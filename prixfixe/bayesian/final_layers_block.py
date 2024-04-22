from functools import partial
import torch
from torch import nn, Generator

from prixfixe.prixfixe import FinalLayersBlock

from pyro.infer.autoguide.utils import deep_getattr, deep_setattr

from pyro.nn import PyroModule
from pyro import poutine
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger, init_to_feasible, init_to_mean
import pyro
import numpy as np

from .TFaffinityScanner import TFaffinityScanner


def init_to_value(site=None, values={}, init_fn=init_to_mean):
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_fn(site)

class BayesianPyroModel(PyroModule):

    def __init__(self, fixed_motifs, n_out, tf_affinity_scanner_kwargs=None):

        super().__init__()

        self.fixed_motifs = fixed_motifs
        self.n_out = n_out


        self.register_buffer('bins', torch.arange(start=0, end=n_out, step=1, requires_grad=False))

        if tf_affinity_scanner_kwargs is None:
            tf_affinity_scanner_kwargs = {}
        self.tf_affinity_scanner_kwargs = tf_affinity_scanner_kwargs

    def forward(self, dna_sequence=None, y_probs=None, y=None):

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
                n_motifs = 100
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

        x_rep = module(dna_sequence)
        x_rep = torch.nn.functional.adaptive_avg_pool1d(x_rep, 1)
        x_rep = x_rep.squeeze(2)
        x_rep = torch.nn.functional.softmax(x_rep, dim=1)
        score = (x_rep * self.bins).sum(dim=1)

        # plates cannot be used with dataloader batches without providing indices
        # var_plate = pyro.plate("var_plate", size=self.n_obs, dim=-2, subsample=idx)

        if y is not None:
            pyro.sample(
                "y",
                pyro.distributions.Normal(
                    score,
                    torch.tensor(1.0, device=score.device),
                ).to_event(2),
                obs=y_probs,
            )

        if y_probs is not None:
            pyro.sample(
                "y_probs",
                pyro.distributions.Categorical(
                    x_rep,
                ).to_event(2),
                obs=y_probs,
            )

        return score


class BaseModule(nn.Module):
    def __init__(
        self,
        model,
        guide_kwargs: dict = None,
        init_loc_fn=init_to_mean(fallback=init_to_feasible),
        guide_class=AutoHierarchicalNormalMessenger,
        **kwargs,
    ):
        """
        Module class which defines AutoGuide given model. Supports multiple model architectures.

        Parameters
        ----------
        kwargs
            arguments for specific model class - e.g. number of genes, values of the prior distribution
        """
        super().__init__()
        self.hist = []

        self._model = model(**kwargs)

        if guide_kwargs is None:
            guide_kwargs = dict()
        if getattr(model, "discrete_variables", None) is not None:
            model = poutine.block(model, hide=model.discrete_variables)
        if issubclass(guide_class, poutine.messenger.Messenger):
            # messenger guides don't need create_plates function
            self._guide = guide_class(
                model,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
            )
        else:
            self._guide = guide_class(
                model,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
                create_plates=self.model.create_plates,
            )
    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def list_obs_plate_vars(self):
        return self.model.list_obs_plate_vars()

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


class BayesianFinalLayersBlock(FinalLayersBlock):
    def __init__(
            self,
            in_channels: int, # for compatibity. Isn't used by block itself
            seqsize: int,  # for compatibity. Isn't used by block itself
            fixed_motifs: np.ndarray = None,
            n_out: int = 18,
    ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)
        self.pyro_module = BaseModule(
            model=BayesianPyroModel,
            fixed_motifs=fixed_motifs,
            n_out=n_out,
            tf_affinity_scanner_kwargs={},
        )

    def forward(self, dna_sequence):
        score = self.pyro_module.model(dna_sequenc=dna_sequence)
        return score

    def train_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        kwargs = {"dna_sequence": batch["x"].to(self.device)}
        if "y_probs" in batch:  # classification
            kwargs["y_probs"] = batch["y_probs"].to(self.device)
        else:  # regression
            kwargs["y"] = batch["y"].to(self.device)
        # pytorch lightning requires a Tensor object for loss
        loss = self.differentiable_loss_fn(
            self.scale_fn(self.pyro_module.model),
            self.scale_fn(self.pyro_module.guide),
            **kwargs
        )
        score = self.pyro_module.model(dna_sequence=kwargs["dna_sequence"], y=None, y_probs=None)
        return score, loss

    def weights_init(self, generator: Generator) -> None:
        pass
