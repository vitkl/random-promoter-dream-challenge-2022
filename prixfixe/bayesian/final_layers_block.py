import contextlib
from functools import partial
import torch
from torch import nn, Generator

from prixfixe.prixfixe import FinalLayersBlock

from pyro.infer.autoguide.utils import deep_getattr, deep_setattr

from pyro.nn import PyroModule
from pyro import poutine
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger, init_to_feasible, init_to_mean
import pyro
from einops import rearrange

from .TFaffinityScanner import TFaffinityScanner


def init_to_value(site=None, values={}, init_fn=init_to_mean):
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_fn(site)

class BayesianPyroModel(PyroModule):

    def __init__(
        self,
        fixed_motifs,
        n_out,
        level2_width=10,
        level3_width=10,
        sigma_prior=1.0,
        use_3_rates=False,
        tf_affinity_scanner_kwargs=None,
        likelihood_scale=1.0,
    ):

        super().__init__()

        self.fixed_motifs = fixed_motifs
        self.n_out = n_out
        self.level2_width = level2_width
        self.level3_width = level3_width
        self.sigma_prior = sigma_prior
        self.use_3_rates = use_3_rates
        if use_3_rates:
            self.n_out = 3

        self.register_buffer('bins', torch.arange(start=0, end=n_out, step=1, requires_grad=False))

        if tf_affinity_scanner_kwargs is None:
            tf_affinity_scanner_kwargs = {}
        self.tf_affinity_scanner_kwargs = tf_affinity_scanner_kwargs

        self.weights = PyroModule()

        self.likelihood_scale = likelihood_scale

    def create_plates(self, dna_sequence=None, y_probs=None, y=None):
        return []  # pyro.plate("obs_plate", size=dna_sequence.shape[0], dim=-3)

    def forward(self, dna_sequence=None, y_probs=None, y=None):

        dna_sequence = rearrange(dna_sequence, "r n b -> r b n")

        name = "simple_tf_effect_model"
        tf_affinity_scanner_mode = "two_layer_conv1d"

        if getattr(self.weights, name, None) is None:
            if self.fixed_motifs is not None:
                motif_length = self.fixed_motifs.shape[-2]
                n_nucleotides = self.fixed_motifs.shape[-1]
                n_motifs = self.fixed_motifs.shape[0]
            else:
                motif_length = 15
                n_nucleotides = 4
                n_motifs = 177
            tf_affinity_scanner_kwargs_ = {
                "name": name,
                "mode": tf_affinity_scanner_mode,
                "fixed_motifs": self.fixed_motifs,
                "n_motifs": n_motifs,
                "motif_length": motif_length,
                "seq_length": dna_sequence.shape[-2],
                "n_nucleotides": n_nucleotides,
                "n_hidden": self.n_out,
                "motif_loc_mean": 0.0,
                "motif_loc_scale": 0.1,
                "motif_weight_alpha": 10.0,
                "pool_fn": torch.nn.MaxPool2d,
                "pool_window": None,
                "return_forward_revcomp": False,
                "use_affinity_scaling_by_motif_complexity": True,
            }
            if self.tf_affinity_scanner_kwargs is not None:
                tf_affinity_scanner_kwargs_.update(self.tf_affinity_scanner_kwargs)
            deep_setattr(
                self.weights,
                name,
                TFaffinityScanner(
                    **tf_affinity_scanner_kwargs_,
                ),
            )
        module = deep_getattr(self.weights, name)
        score = module(
            dna_sequence.unsqueeze(-3),
            width=self.level2_width,
            level3_width=self.level3_width
        )
        # print("x_rep mean", x_rep.mean())
        # print("x_rep min", x_rep.min())
        # print("x_rep max", x_rep.max())
        # sum across positions
        # x_rep = torch.nn.functional.adaptive_avg_pool1d(x_rep, 1).squeeze(-1)
        # x_re = x_rep.sum(dim=-1)
        # print("x_rep avg pool mean", x_rep.mean())
        # print("x_rep avg pool min", x_rep.min())
        # print("x_rep avg pool max", x_rep.max())

        # the prior for bin approach probably has to put most of the probability mass on the first few bins
        # x_re = x_re + self.bins.flip(-1) / 5
        # x_re = torch.nn.functional.softmax(x_re, dim=-1)
        #print("x_re mean", x_re.mean(-2))
        #print("x_re min", x_re.min(-2))
        #print("x_re max", x_re.max(-2))
        #score = (x_re * self.bins).sum(dim=-1)

        # Alternative: use softplus to make the scores positive
        # and to have a smooth transition across detected values
        #score = torch.nn.functional.softplus(
        #    x_re - torch.tensor(2.0, device=x_re.device)
        #).squeeze(-1) / torch.tensor(0.7, device=x_re.device)
        #if self.use_3_rates:
        #    score = score[:, 0] / (score[:, 1] + score[:, 2])
        #print("score mean", score.mean())
        #print("score min", score.min())
        #print("score max", score.max())

        # plates cannot be used with dataloader batches without providing indices (idx)
        # var_plate = pyro.plate("var_plate", size=self.n_obs, dim=-2, subsample=idx)

        sigma = pyro.sample(
            "sigma",
            pyro.distributions.Exponential(
                torch.tensor(self.sigma_prior, device=score.device).expand([1, 1])
            ).to_event(2)
        )

        if not self.training:
            pyro.deterministic("y_pred", score)

        if y is not None:
            # loss scaling to account for minibatch size
            likelihood_scale_context = (
                pyro.poutine.scale(
                    scale=torch.tensor(self.likelihood_scale / len(y), device=y.device)
                )
                if self.likelihood_scale != 1.0
                else contextlib.nullcontext()
            )
            with likelihood_scale_context:
                pyro.sample(
                    "y",
                    pyro.distributions.Normal(
                        score,
                        sigma,
                    ).to_event(2),
                    obs=y,
                )

        # if y_probs is not None:
        if False:
            pyro.sample(
                "y_probs",
                pyro.distributions.Multinomial(
                    probs=x_rep,
                ).to_event(1),
                obs=(y_probs > torch.tensor(0.5, device=y_probs.device)).float(),
            )

        return score


class BayesianFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int,  # for compatibity. Isn't used by block itself
        seqsize: int,  # for compatibity. Isn't used by block itself
        model=BayesianPyroModel,
        guide_kwargs: dict = None,
        init_loc_fn=init_to_mean(fallback=init_to_feasible),
        guide_class=AutoHierarchicalNormalMessenger,
        scale_elbo=1.0,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)

        self._model = model(**kwargs)
        if getattr(self._model, "discrete_variables", None) is not None:
            self._model = poutine.block(self._model, hide=model.discrete_variables)

        if guide_kwargs is None:
            guide_kwargs = dict()
        if issubclass(guide_class, poutine.messenger.Messenger):
            # messenger guides don't need create_plates function
            self._guide = guide_class(
                self.model,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
            )
        else:
            self._guide = guide_class(
                self.model,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
                create_plates=self.model.create_plates,
            )

        self.loss_fn = pyro.infer.Trace_ELBO()
        self.differentiable_loss_fn = self.loss_fn.differentiable_loss
        self.scale_elbo = scale_elbo
        self.scale_fn = (
            lambda obj: pyro.poutine.scale(obj, self.scale_elbo) if self.scale_elbo != 1.0 else obj
        )

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    def init_to_value(self, site):
        if getattr(self.model, "np_init_vals", None) is not None:
            init_vals = {
                k: getattr(self.model, f"init_val_{k}")
                for k in self.model.np_init_vals.keys()
            }
        else:
            init_vals = dict()
        return init_to_value(
            site=site,
            values=init_vals,
            # init_fn=partial(init_to_median, num_samples=501, fallback=init_to_mean),
            init_fn=init_to_mean,
        )

    def forward(self, dna_sequence):
        score = self.model(dna_sequence=dna_sequence)
        # score = self.model.final.guide.median(dna_sequence=dna_sequence)["y_pred"]
        return score

    def train_step(self, batch):
        """Training step for Pyro training."""
        kwargs = {"dna_sequence": batch["x"].to(self.device)}
        if "y_probs" in batch:  # classification
            kwargs["y_probs"] = batch["y_probs"].to(self.device)
        else:  # regression
            kwargs["y"] = batch["y"].to(self.device)
        # pytorch lightning requires a Tensor object for loss
        loss = self.differentiable_loss_fn(
            self.scale_fn(self.model),
            self.scale_fn(self.guide),
            **kwargs
        )
        score = self.model(dna_sequence=kwargs["dna_sequence"], y=None, y_probs=None)
        return score, loss

    def weights_init(self, generator: Generator) -> None:
        pass
