from typing import Optional

import einops
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from einops import rearrange
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroSample
from torch.nn import Identity



class StochasticShift(nn.Module):
    """
    Stochastically shift a one hot encoded DNA sequence.
    adapted from https://github.com/calico/scBasset/blob/main/scbasset/basenji_utils.py
    """

    def __init__(self, shift_max: int = 3, pad: float = 0.25):
        super().__init__()
        self.shift_max = shift_max

        self.register_buffer("pad", torch.tensor(pad))

    def forward(self, seq_1hot: torch.Tensor):
        if not self.training:
            return seq_1hot
        shifts = torch.randint(
            low=-self.shift_max,
            high=self.shift_max + 1,
            size=(seq_1hot.shape[0],),  # first dim is the batch dim
        )
        return torch.stack(
            [
                shift_seq(seq, shift, pad=self.pad)
                for seq, shift in zip(seq_1hot, shifts)
            ]
        )


def shift_seq(
    seq: torch.Tensor,  # input shape: (1, seq_length, n_nucleotides)
    shift: int,
    pad: float = 0.0,
):
    """Shift a sequence left or right by shift_amount.
    adapted from https://github.com/calico/scBasset/blob/main/scbasset/basenji_utils.py
    Args:
    seq: [batch_size, seq_depth, seq_length] sequence
    shift: signed shift value (torch.tensor)
    pad: value to fill the padding (float)
    """

    # if no shift return the sequence
    if shift == 0:
        return seq

    # create the padding
    pad = pad * torch.ones_like((seq[:, : abs(shift), :]))

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift, :]
        # cat to the left along the sequence axis
        return torch.cat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        # cat to the right along the sequence axis
        return torch.cat([sliced_seq, pad], axis=1)

    if shift > 0:  # if shift is positive shift_right
        return _shift_right(seq)
    # if shift is negative shift_left
    return _shift_left(seq)



class Exp(torch.nn.Module):
    """
    Exponential activation function
    (from Koo, 2021, PMID: 34322657)
    """

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class TFaffinityScanner(PyroModule):

    use_refined_motifs_below_zero_beta = 5.0
    name_prefix = ""

    def __init__(
        self,
        name: str = "",
        fixed_motifs: Optional[np.array] = None,
        n_motifs: int = None,
        n_binding_modes: int = 1,
        use_motif_refining: bool = True,
        mode: str = "one_layer_conv2d",
        use_refined_motifs_below_zero: bool = False,
        motif_length: int = 5,
        seq_length: int = 7,
        n_nucleotides: int = 4,
        motif_loc_mean: float = 0.0,
        motif_loc_scale: float = 0.01,
        motif_weight_alpha: float = 100.0,
        n_hidden: int = 128,
        activation_fn: torch.nn.Module = Exp,
        pool_fn: torch.nn.Module = torch.nn.AvgPool2d,
        pool_window: Optional[int] = None,
        pool_stride: Optional[int] = None,
        use_reverse_complement: bool = False,
        return_forward_revcomp: bool = False,
        shift_seq_val: int = 0,
        use_affinity_scaling_by_motif_complexity: bool = True,
        use_affinity_scaling_by_motif_complexity_detach: bool = False,
        padding: str = "same",
        n_layers: int = 3,
        use_sqrt_normalisation: bool = True,
        use_dilation: bool = False,
        use_non_residual_dilation: bool = False,
        use_full_dilated_layer: bool = False,
        use_competition_normalisation: bool = False,
        use_hill_function: bool = False,
        dilation: int = 8,
        use_normal_prior: bool = False,
        use_horseshoe_prior: bool = False,
    ):
        super().__init__()
        if fixed_motifs is None and n_motifs is None:
            raise ValueError("Must specify either fixed_motifs or num_motifs")
        if fixed_motifs is not None:
            n_motifs = fixed_motifs.shape[0]

        # save properties
        self.name = name
        self.mode = mode
        self.n_motifs = n_motifs
        self.n_binding_modes = n_binding_modes
        self.use_motif_refining = use_motif_refining
        self.use_refined_motifs_below_zero = use_refined_motifs_below_zero
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.n_nucleotides = n_nucleotides
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.use_sqrt_normalisation = use_sqrt_normalisation
        self.use_dilation = use_dilation
        self.use_non_residual_dilation = use_non_residual_dilation
        self.use_full_dilated_layer = use_full_dilated_layer
        self.use_competition_normalisation = use_competition_normalisation
        self.use_hill_function = use_hill_function
        self.dilation = dilation
        self.use_normal_prior = use_normal_prior
        self.use_horseshoe_prior = use_horseshoe_prior

        self.complement = {
            "A": "T",
            "C": "G",
            "G": "C",
            "T": "A",
        }
        self.complement_index = [
            np.where(v == np.array(list(self.complement.keys())))[0][0]
            for k, v in self.complement.items()
        ]
        self.use_reverse_complement = use_reverse_complement
        self.return_forward_revcomp = return_forward_revcomp
        self.use_affinity_scaling_by_motif_complexity = (
            use_affinity_scaling_by_motif_complexity
        )
        self.use_affinity_scaling_by_motif_complexity_detach = (
            use_affinity_scaling_by_motif_complexity_detach
        )
        self.shift_fn = (
            StochasticShift(shift_seq_val) if shift_seq_val > 0 else Identity()
        )
        self.padding = padding

        # Set up activation and pooling
        self.activation1 = activation_fn()
        self.pool_window = pool_window
        pool_fn_kwargs = dict()
        if pool_fn is torch.nn.AvgPool2d:
            pool_fn_kwargs["divisor_override"] = 1
        if pool_window is None:
            pool_window = seq_length - motif_length + 1
            self.pool_window = pool_window
            self.pool1 = pool_fn((pool_window, 1), **pool_fn_kwargs)
        else:
            if pool_stride is None:
                pool_stride = pool_window
            self.pool1 = pool_fn((pool_window, 1), (pool_stride, 1), **pool_fn_kwargs)
            self.pool_window = pool_window
            self.pool_stride = pool_stride
            self.seq_length_after_pooling = int(
                (
                    (self.seq_length - (self.motif_length - 1))
                    - (self.pool_window - 1)
                    - 1
                )
                / self.pool_stride
                + 1
            )

        # Define the prior for convolutional weights
        self.register_buffer(
            "motif_loc_mean",
            torch.tensor(float(motif_loc_mean)),
        )
        self.register_buffer(
            "motif_loc_scale",
            torch.tensor(float(motif_loc_scale)),
        )
        self.register_buffer(
            "motif_weight_alpha",
            torch.tensor(float(motif_weight_alpha)),
        )
        self.weights = PyroModule()
        self.fixed_motifs = fixed_motifs
        if fixed_motifs is not None:
            assert fixed_motifs.shape == (
                self.n_motifs,
                1,
                self.motif_length,
                self.n_nucleotides,
            )
            self.register_buffer(
                "fixed_motifs_tensor",
                torch.tensor(fixed_motifs.astype("float32")),
            )
        self.register_buffer("ones", torch.tensor(float(1.0)))
        self.register_buffer("zeros", torch.tensor(float(0.0)))

    def create_horseshoe_prior(
        self,
        name,
        weights_shape,
        weights_prior_scale=None,
        weights_prior_tau=None,
        scale_distribution=dist.HalfNormal,  # TODO figure out which distribution to use HalfCauchy has mean=Inf so can't use it
    ):
        # Create scalar tau (like sd for horseshoe prior) =====================
        tau_name = f"{name}tau"
        if getattr(self.weights, tau_name, None) is None:
            if weights_prior_tau is None:
                weights_prior_tau = self.weights_prior_tau
            if getattr(self, f"{tau_name}_scale", None) is None:
                self.register_buffer(f"{tau_name}_scale", weights_prior_tau)
            deep_setattr(
                self.weights,
                tau_name,
                PyroSample(
                    lambda prior: scale_distribution(
                        getattr(self, f"{tau_name}_scale"),
                    )
                    .expand([1])
                    .to_event(1),
                ),
            )
        tau = deep_getattr(self.weights, tau_name)

        # Create weights (like mean for horseshoe prior) =====================
        weights_name = f"{name}weights"
        if getattr(self.weights, weights_name, None) is None:
            deep_setattr(
                self.weights,
                weights_name,
                PyroSample(
                    lambda prior: dist.Normal(
                        self.zeros,
                        self.ones,
                    )
                    .expand(weights_shape)
                    .to_event(len(weights_shape)),
                ),
            )
        unscaled_weights = deep_getattr(self.weights, weights_name)

        if getattr(self, "use_gamma_horseshoe_prior", False):
            # Create elementwise lambdas using Gamma distribution (like sd for horseshoe prior) =====================
            lambdas_name = f"{name}lambdas"
            if getattr(self.weights, lambdas_name, None) is None:
                if weights_prior_scale is None:
                    weights_prior_scale = self.weights_prior_scale
                if getattr(self, f"{lambdas_name}_scale", None) is None:
                    self.register_buffer(f"{lambdas_name}_scale", weights_prior_scale)
                deep_setattr(
                    self.weights,
                    lambdas_name,
                    PyroSample(
                        lambda prior: dist.Gamma(
                            tau,
                            getattr(self, f"{lambdas_name}_scale"),
                        )
                        .expand(weights_shape)
                        .to_event(len(weights_shape)),
                    ),
                )
            lambdas = deep_getattr(self.weights, lambdas_name)
        else:
            # Create elementwise lambdas (like sd for horseshoe prior) =====================
            lambdas_name = f"{name}lambdas"
            if getattr(self.weights, lambdas_name, None) is None:
                if weights_prior_scale is None:
                    weights_prior_scale = self.weights_prior_scale
                if getattr(self, f"{lambdas_name}_scale", None) is None:
                    self.register_buffer(f"{lambdas_name}_scale", weights_prior_scale)
                deep_setattr(
                    self.weights,
                    lambdas_name,
                    PyroSample(
                        lambda prior: scale_distribution(
                            getattr(self, f"{lambdas_name}_scale"),
                        )
                        .expand(weights_shape)
                        .to_event(len(weights_shape)),
                    ),
                )
            lambdas = deep_getattr(self.weights, lambdas_name)
            lambdas = tau * lambdas

        weights = lambdas * unscaled_weights
        if not self.training:
            pyro.deterministic(f"{self.name_prefix}{name}", weights)
        return weights

    def _get_motif_weights(
        self,
        name="",
        use_prior=True,
        prior_alpha=None,
        n_binding_modes=1,
        use_refined_motifs_below_zero=None,
        n_nucleotides=None,
    ):

        if n_nucleotides is None:
            n_nucleotides = self.n_nucleotides

        if use_refined_motifs_below_zero is None:
            use_refined_motifs_below_zero = self.use_refined_motifs_below_zero

        weights_prior = dist.SoftLaplace
        if self.use_normal_prior:
            weights_prior = dist.Normal

        if self.use_motif_refining:
            # Motif weights
            weights_name = f"{name}motif_loc"
            if getattr(self.weights, weights_name, None) is None:
                deep_setattr(
                    self.weights,
                    weights_name,
                    PyroSample(
                        lambda prior: weights_prior(
                            self.motif_loc_mean,
                            self.motif_loc_scale,
                        )
                        .expand(
                            [
                                self.n_motifs * n_binding_modes,
                                1,
                                self.motif_length,
                                n_nucleotides,
                            ]
                        )  # [h, 1, p, n]
                        .to_event(4),
                    ),
                )
            motif_loc = deep_getattr(self.weights, weights_name)
            weights_name = f"{name}motif_weight"
            if (self.fixed_motifs is not None) and use_prior:
                if prior_alpha is None:
                    if getattr(self.weights, weights_name, None) is None:
                        deep_setattr(
                            self.weights,
                            weights_name,
                            PyroSample(
                                lambda prior: dist.Gamma(
                                    self.motif_weight_alpha,
                                    self.motif_weight_alpha,
                                )
                                .expand(
                                    [
                                        self.n_motifs * n_binding_modes,
                                        1,
                                        self.motif_length,
                                        n_nucleotides,
                                    ]
                                )
                                .to_event(4),
                            ),
                        )
                else:
                    if getattr(self, f"{name}motif_weight_alpha", None) is None:
                        self.register_buffer(
                            f"{name}motif_weight_alpha", torch.tensor(prior_alpha)
                        )
                    if getattr(self.weights, weights_name, None) is None:
                        deep_setattr(
                            self.weights,
                            weights_name,
                            PyroSample(
                                lambda prior: dist.Gamma(
                                    self.motif_weight_alpha,
                                    self.motif_weight_alpha,
                                )
                                .expand(
                                    [
                                        self.n_motifs * n_binding_modes,
                                        1,
                                        self.motif_length,
                                        n_nucleotides,
                                    ]
                                )
                                .to_event(4),
                            ),
                        )

                motif_weight = deep_getattr(self.weights, weights_name)
            # get the motif weights
            motifs = motif_loc
            if (self.fixed_motifs is not None) and use_prior:
                if n_binding_modes > 1:
                    fixed_motifs_tensor = torch.cat(
                        [self.fixed_motifs_tensor for _ in range(n_binding_modes)], -4
                    )
                    motifs = motifs + fixed_motifs_tensor * motif_weight
                else:
                    motifs = motifs + self.fixed_motifs_tensor * motif_weight
        else:
            motifs = self.fixed_motifs_tensor

        if use_refined_motifs_below_zero:
            motifs = -torch.nn.functional.softplus(
                -motifs - torch.tensor(1.0, device=motifs.device),
                beta=self.use_refined_motifs_below_zero_beta,
            )
        if not self.training:
            # track refined motifs
            pyro.deterministic(f"{self.name}{name}refined_motifs", motifs)
        if (self.fixed_motifs is not None) and use_prior and (not self.training):
            # track initial motifs
            init_motifs = self.fixed_motifs_tensor
            if use_refined_motifs_below_zero:
                init_motifs = -torch.nn.functional.softplus(
                    -self.fixed_motifs_tensor - torch.tensor(1.0, device=motifs.device),
                    beta=5.0,
                )
            pyro.deterministic(f"{self.name}{name}initial_motifs", init_motifs)

        return motifs

    def _get_positive_weights(self, shape, name="", prior_alpha=None):
        if prior_alpha is None:
            prior_alpha = 1.0
        if getattr(self, f"{name}prior_alpha", None) is None:
            self.register_buffer(f"{name}prior_alpha", torch.tensor(np.array(prior_alpha).astype("float32")))
        # Motif weights
        weights_name = f"{name}weight"
        if getattr(self.weights, weights_name, None) is None:
            deep_setattr(
                self.weights,
                weights_name,
                PyroSample(
                    lambda prior: dist.Gamma(
                        getattr(self, f"{name}prior_alpha"),
                        getattr(self, f"{name}prior_alpha"),
                    )
                    .expand(shape)
                    .to_event(len(shape)),
                ),
            )
        return deep_getattr(self.weights, weights_name)

    def _get_weights(self, shape, name="", prior_mean=None, prior_sigma=None):
        if prior_mean is None:
            prior_mean = 0.0
        if getattr(self, f"{name}prior_mean", None) is None:
            self.register_buffer(f"{name}prior_mean", torch.tensor(np.array(prior_mean).astype("float32")))
        if prior_sigma is None:
            prior_sigma = 1.0
        if getattr(self, f"{name}prior_sigma", None) is None:
            self.register_buffer(f"{name}prior_sigma", torch.tensor(np.array(prior_sigma).astype("float32")))
        if getattr(self, f"{name}prior_tau", None) is None:
            self.register_buffer(f"{name}prior_tau", torch.tensor(np.array(1.0).astype("float32")))
        if getattr(self, f"{name}motif_length_tensor", None) is None:
            self.register_buffer(
                f"{name}motif_length_tensor", torch.tensor(np.array(self.motif_length).astype("float32"))
            )
        if (prior_mean == 0.0) and self.use_horseshoe_prior:
            return self.create_horseshoe_prior(
                name=name,
                weights_shape=shape,
                weights_prior_scale=getattr(self, f"{name}prior_sigma", None),
                weights_prior_tau=getattr(self, f"{name}prior_tau", None),
                scale_distribution=dist.HalfNormal,
            )
        # Motif weights
        weights_prior = dist.SoftLaplace
        if self.use_normal_prior:
            weights_prior = dist.Normal
        weights_name = f"{name}laplace_weight"
        if getattr(self.weights, weights_name, None) is None:
            deep_setattr(
                self.weights,
                weights_name,
                PyroSample(
                    lambda prior: weights_prior(
                        getattr(self, f"{name}prior_mean"),
                        getattr(self, f"{name}prior_sigma"),
                    )
                    .expand(shape)
                    .to_event(len(shape)),
                ),
            )
        return deep_getattr(self.weights, weights_name)

    def _apply_conv2d(self, x, motifs, padding="same"):
        # Do convolution
        # input shape: (n_regions, 1, seq_length, n_nucleotides)
        if (
            self.use_reverse_complement
            or self.return_forward_revcomp
            or (not self.training)
        ):
            reverse_complement = torch.nn.functional.conv2d(
                x,
                motifs[:, :, :, self.complement_index].flip(-2),
                padding=padding,
            )
        else:
            reverse_complement = None
        x = torch.nn.functional.conv2d(x, motifs, padding=padding)
        # output shape: (n_regions, n_motifs, seq_length - (motif_length - 1), 1)
        return x, reverse_complement

    def forward(self, x, **kwargs):
        return getattr(self, self.mode)(x, **kwargs)

    def compute_motif_complexity_2d(
        self, motifs_tensor: torch.Tensor, n_binding_modes=1
    ):
        if n_binding_modes == 1:
            # sum across nucleotides, sum across positions
            # [h, 1, p, n]
            return motifs_tensor.abs().sum(-1).sum(-1).sum(-1)
        else:
            motifs_tensor = rearrange(
                motifs_tensor,
                "(h m) q p n -> h m q p n",  # [h, 1, p, n]
                h=self.n_motifs,
                m=self.n_binding_modes,
            )
            # sum across nucleotides, sum across positions, sum across binding modes
            return motifs_tensor.abs().sum(-1).sum(-1).sum(-1).sum(-1)

    def hill_function(self, x, ka, n):
        y = (
            torch.ones((), device=x.device)
            / (
                torch.ones((), device=x.device)
                # prevent division by zero
                + (ka / (x + torch.tensor(1e-6, device=x.device))) ** n
                # keep 0s as 0s
            )
        ) * (x.detach() > torch.zeros((), device=x.device)).float()

        return y

    def compute_motif_complexity_1d(
        self, motifs_tensor: torch.Tensor, n_binding_modes=1
    ):
        if n_binding_modes == 1:
            # sum across nucleotides, sum across positions
            # [h, n, p]
            return motifs_tensor.abs().sum(-1).sum(-1)
        else:
            motifs_tensor = rearrange(
                motifs_tensor,
                "(h m) n p -> h m n p",  # [h, p, n]
                h=self.n_motifs,
                m=self.n_binding_modes,
            )
            # sum across nucleotides, sum across positions, sum across binding modes
            return motifs_tensor.abs().sum(-1).sum(-1).sum(-1)

    def one_layer_conv2d(self, x):
        # Layer 1 - learning TF-DNA motif  ===============================
        # shift sequence
        x = self.shift_fn(x)
        # get motif weights
        motifs = self._get_motif_weights(use_prior=True)

        # Do convolution
        x, reverse_complement = self._apply_conv2d(x, motifs, padding=self.padding)

        # Apply activation
        x = self.activation1(x)
        if self.use_reverse_complement and not self.return_forward_revcomp:
            x = x + self.activation1(reverse_complement)
        elif self.return_forward_revcomp:
            x = torch.cat([x, self.activation1(reverse_complement)], dim=-1)

        # apply pooling
        x = self.pool1(x)

        # Output sizes If max-pooling across the entire sequence:
        # [regions, motifs, position] -> [regions, motifs]
        # or [regions, motifs, position, forward/revcomp] -> [regions, motifs, forward/revcomp]
        # scale affinity by motif complexity
        x = x.squeeze(-1).squeeze(-1)
        if self.use_affinity_scaling_by_motif_complexity:
            motif_complexity = self.compute_motif_complexity_2d(
                motifs
                if not self.use_affinity_scaling_by_motif_complexity_detach
                else motifs.detach(),
                n_binding_modes=self.n_binding_modes,
            ) / torch.tensor(50.0, device=motifs.device)
            if self.return_forward_revcomp:
                x = torch.einsum("rhpo,h->rhpo", x, motif_complexity)
            else:
                # print("motif_complexity mean", motif_complexity.mean())
                x = torch.einsum("rhp,h->rhp", x, motif_complexity)
        else:
            x = x / torch.tensor(50.0, device=motifs.device)
        return x

    def residual_layer(self, x1_rhp, region_motif_coo_rhp: torch.Tensor = None, width: int = 10, layer=2):
        n_motifs = x1_rhp.shape[-2]
        scaling = 1.0
        if self.use_sqrt_normalisation:
            scaling = 1.0 / np.sqrt(n_motifs * width)
        weights = self._get_weights(
            shape=[n_motifs, n_motifs, width],
            name=f"tn5_layer{layer}",
            prior_mean=0.0,
            prior_sigma=scaling,
        )
        weights = (
            weights + weights.flip(-1)
        ) / torch.tensor(2.0, device=weights.device)
        residual_weight = self._get_positive_weights(
            shape=[1, n_motifs, 1],
            name=f"tn5_layer{layer}_residual",
            prior_alpha=2.0,
        )
        # {N, in, L} & {out, in, L} + residual
        x_rhp = torch.nn.functional.conv1d(x1_rhp, weights, padding="same")
        x_rhp = x_rhp / torch.tensor(5.0, device=x_rhp.device)
        x_rhp = torch.nn.functional.softplus(x_rhp) * x1_rhp
        if region_motif_coo_rhp is not None:
            x_rhp = torch.concat([x_rhp, x1_rhp * residual_weight], dim=-2)
        else:
            x_rhp = (x_rhp + x1_rhp * residual_weight) / torch.tensor(2.0, device=x_rhp.device)
        # print(f"x_rhp {layer} mean", x_rhp.mean())
        # print(f"x_rhp {layer} min", x_rhp.min())
        # print(f"x_rhp {layer} max", x_rhp.max())
        return x_rhp

    def dilated_layer(
        self,
        x1_rhp: torch.Tensor,
        x_sums_rhp: torch.Tensor,
        width: int = 3,
        dilation: int = 1,
        layer=2,
    ):
        n_motifs = x1_rhp.shape[-2]
        scaling = 1.0
        if self.use_sqrt_normalisation:
            scaling = 1.0 / np.sqrt(n_motifs * width)
        weights = self._get_weights(
            shape=[n_motifs, n_motifs, width],
            name=f"tn5_layer{layer}",
            prior_mean=0.0,
            prior_sigma=scaling,
        )
        weights = (
            weights + weights.flip(-1)
        ) / torch.tensor(2.0, device=weights.device)
        dilated_weight = self._get_positive_weights(
            shape=[n_motifs, 1, width],
            name=f"tn5_layer{layer}_dilated",
            prior_alpha=1.0,
        )
        dilated_weight = (
            dilated_weight + dilated_weight.flip(-1)
        ) / torch.tensor(2.0, device=dilated_weight.device)
        residual_weight = self._get_positive_weights(
            shape=[1, n_motifs, 1],
            name=f"tn5_layer{layer}_residual",
            prior_alpha=2.0,
        )
        # {N, in, L} & {out, in, L} + residual
        x_sums_rhp = torch.nn.functional.conv1d(
            x_sums_rhp, dilated_weight,
            groups=n_motifs,
            dilation=dilation,
            padding="same"
        ) / torch.tensor(2.0, device=x_sums_rhp.device) # / torch.tensor(float(dilation), device=x_sums_rhp.device)
        x_rhp = torch.nn.functional.conv1d(
            x_sums_rhp, weights,
            padding="same"
        )
        x_rhp = x_rhp / torch.tensor(5.0, device=x_rhp.device)
        x_rhp = torch.nn.functional.softplus(x_rhp) * x1_rhp
        x_rhp = (x_rhp + x1_rhp * residual_weight) / torch.tensor(2.0, device=x_rhp.device)
        # print(f"x_rhp {layer} mean", x_rhp.mean())
        # print(f"x_rhp {layer} min", x_rhp.min())
        # print(f"x_rhp {layer} max", x_rhp.max())
        return x_rhp, x_sums_rhp

    def full_dilated_layer(
        self,
        x1_rhp: torch.Tensor,
        width: int = 3,
        dilation: int = 7,
        layer=2,
    ):
        if self.return_forward_revcomp:
            n_motifs = x1_rhp.shape[-3]
        else:
            n_motifs = x1_rhp.shape[-2]
        scaling = 1.0
        if self.use_sqrt_normalisation:
            scaling = 1.0 / np.sqrt(n_motifs * width)
        residual_weight = self._get_positive_weights(
            shape=[1, n_motifs, 1] if not self.return_forward_revcomp else [1, n_motifs, 1, 1],
            name=f"tn5_layer{layer}_residual",
            prior_alpha=2.0,
        )
        x_rhp = torch.zeros(x1_rhp.shape, device=x1_rhp.device)
        x_sums_rhp = x1_rhp
        for i in range(dilation):
            i = i + 1
            weights = self._get_weights(
                shape=[n_motifs, n_motifs, 1]
                if not self.return_forward_revcomp else [2 * n_motifs, 2 * n_motifs, 1],
                name=f"tn5_layer{layer}_dilation{i}",
                prior_mean=0.0,
                prior_sigma=scaling,
            )
            dilated_weight = self._get_positive_weights(
                shape=[n_motifs, 1, width]
                if not self.return_forward_revcomp else [n_motifs, 1, width],
                name=f"tn5_layer{layer}_dilated_dilation{i}",
                prior_alpha=1.0,
            )
            if not self.return_forward_revcomp:
                dilated_weight = (
                    dilated_weight + dilated_weight.flip(-1)
                ) / torch.tensor(2.0, device=x_rhp.device)
                # {N, in, L} & {out, in, L} + residual
                x_sums_rhp = torch.nn.functional.conv1d(
                    x_sums_rhp, dilated_weight,
                    groups=n_motifs,
                    dilation=2 ** (i - 1),
                    padding="same"
                ) / torch.tensor(2.0, device=x1_rhp.device)
                x_rhp = x_rhp + torch.nn.functional.conv1d(
                    x_sums_rhp, weights,
                    padding="same",
                )
            else:
                # {N, in, L} & {out, in, L} + residual
                x_sums_rhp_reverse = torch.nn.functional.conv1d(
                    x_sums_rhp[:, :, :, 1], dilated_weight.flip(-1),
                    groups=n_motifs,
                    dilation=2 ** (i - 1),
                    padding="same"
                ) / torch.tensor(2.0, device=x1_rhp.device)
                x_sums_rhp_forward = torch.nn.functional.conv1d(
                    x_sums_rhp[:, :, :, 0], dilated_weight,
                    groups=n_motifs,
                    dilation=2 ** (i - 1),
                    padding="same"
                ) / torch.tensor(2.0, device=x1_rhp.device)
                x_sums_rhp = torch.stack([x_sums_rhp_forward, x_sums_rhp_reverse], dim=-1)
                x_sums_rhp = einops.rearrange(x_sums_rhp, "r h p f -> r (h f) p", f=2, h=n_motifs)
                x_rhp_ = torch.nn.functional.conv1d(
                    x_sums_rhp, weights,
                    padding="same",
                )
                x_rhp_ = einops.rearrange(x_rhp_, "r (h f) p -> r h p f", f=2, h=n_motifs)
                x_sums_rhp = einops.rearrange(x_sums_rhp, "r (h f) p -> r h p f", f=2, h=n_motifs)
                x_rhp = x_rhp + x_rhp_
        x_rhp = x_rhp / torch.tensor(10.0, device=x_rhp.device)
        x_rhp = torch.nn.functional.softplus(x_rhp) * x1_rhp
        x_rhp = (x_rhp + x1_rhp * residual_weight) / torch.tensor(2.0, device=x_rhp.device)
        # print(f"x_rhp {layer} mean", x_rhp.mean())
        # print(f"x_rhp {layer} min", x_rhp.min())
        # print(f"x_rhp {layer} max", x_rhp.max())
        return x_rhp

    def dilated_non_residual_layer(
        self,
        x1_rhp: torch.Tensor,
        width: int = 3,
        dilation: int = 1,
        layer=2,
    ):
        n_motifs = x1_rhp.shape[-2]
        scaling = 1.0
        if self.use_sqrt_normalisation:
            scaling = 1.0 / np.sqrt(n_motifs * width)
        weights = self._get_weights(
            shape=[n_motifs, n_motifs, width],
            name=f"tn5_layer{layer}",
            prior_mean=0.0,
            prior_sigma=scaling,
        )
        weights = (
            weights + weights.flip(-1)
        ) / torch.tensor(2.0, device=weights.device)
        # dilated_weight = self._get_positive_weights(
        #    shape=[n_motifs, 1, width],
        #    name=f"tn5_layer{layer}_dilated",
        #    prior_alpha=1.0,
        # )
        residual_weight = self._get_positive_weights(
            shape=[1, n_motifs, 1],
            name=f"tn5_layer{layer}_residual",
            prior_alpha=2.0,
        )
        # {N, in, L} & {out, in, L} + residual
        # x_sums_rhp = torch.nn.functional.conv1d(
        #    x_sums_rhp, dilated_weight,
        #    groups=n_motifs,
        #    dilation=dilation,
        #    padding="same"
        # ) / torch.tensor(2.0, device=x_sums_rhp.device) # / torch.tensor(float(dilation), device=x_sums_rhp.device)
        x_rhp = torch.nn.functional.conv1d(
            x1_rhp, weights,
            dilation=dilation,
            padding="same"
        )
        x_rhp = x_rhp / torch.tensor(1.0, device=x_rhp.device)
        x_rhp = torch.nn.functional.softplus(x_rhp) #  * x1_rhp
        x_rhp = (x_rhp + x1_rhp * residual_weight) / torch.tensor(1.0, device=x_rhp.device)
        # print(f"x_rhp {layer} mean", x_rhp.mean())
        # print(f"x_rhp {layer} min", x_rhp.min())
        # print(f"x_rhp {layer} max", x_rhp.max())
        return x_rhp

    def two_layer_conv1d(
        self, x, region_motif_coo_rhp: torch.Tensor = None, width: int = 10, level3_width: int = 10,
    ):
        # Layer 1 - learning Tn5-DNA motif  ===============================
        x1_rhp = self.one_layer_conv2d(x)
        #if self.return_forward_revcomp:
        #    x1_rhp = einops.rearrange(x1_rhp, "r h p f -> r (h f) p", f=2, h=self.n_motifs)
        if region_motif_coo_rhp is not None:
            x1_rhp = torch.concat(
                [
                    x1_rhp,
                    region_motif_coo_rhp,
                ],
                dim=-2,
            )
        x1_rhp = x1_rhp / torch.tensor(5.0, device=x1_rhp.device)
        # print("x1_rhp 1 mean", x1_rhp.mean())
        # print("x1_rhp 1 min", x1_rhp.min())
        # print("x1_rhp 1 max", x1_rhp.max())

        # Layer 2 - learning interactions between Tn5 sites =============== dilated_non_residual_layer
        if self.use_dilation:
            i = 2
            x_rhp, x_sums_rhp = self.dilated_layer(
                x1_rhp, x1_rhp, width=width, layer=i, dilation=1,
            )
            i = i + 1
            for _ in range(self.n_layers - 1):
                x_rhp, x_sums_rhp = self.dilated_layer(
                    x_rhp, x_sums_rhp, width=width, layer=i, dilation=2 ** (i - 2),
                )
                i = i + 1
        elif self.use_non_residual_dilation:
            i = 2
            x_rhp = self.dilated_non_residual_layer(
                x1_rhp, width=width, layer=i, dilation=1,
            )
            i = i + 1
            for _ in range(self.n_layers - 1):
                x_rhp = self.dilated_non_residual_layer(
                    x_rhp, width=width, layer=i, dilation=2 ** (i - 2),
                )
                i = i + 1
        elif self.use_full_dilated_layer:
            i = 2
            x_rhp = self.full_dilated_layer(
                x1_rhp, width=width, layer=i, dilation=self.dilation,
            )
            i = i + 1
            for _ in range(self.n_layers - 1):
                x_rhp = self.full_dilated_layer(
                    x_rhp, width=width, layer=i, dilation=self.dilation,
                )
                i = i + 1
        else:
            i = 2
            x_rhp = self.residual_layer(x1_rhp, region_motif_coo_rhp, width=width, layer=i)
            i = i + 1
            for _ in range(self.n_layers - 1):
                x_rhp = self.residual_layer(x_rhp, region_motif_coo_rhp, width=width, layer=i)
                i = i + 1
        # print("x1_rhp 1 mean", x1_rhp.mean())
        # print("x1_rhp 1 min", x1_rhp.min())
        # print("x1_rhp 1 max", x1_rhp.max())

        if self.return_forward_revcomp:
            x_rhp = x_rhp.sum(dim=-1)

        if self.use_competition_normalisation:
            x_rhp = x_rhp / (x_rhp.sum(-2, keepdim=True) + torch.tensor(1.0, device=x_rhp.device))

        if not self.use_hill_function:
            # Layer 3 - summing up effects per batch per position ===============
            n_motifs = x_rhp.shape[-2]
            scaling = 1.0
            if self.use_sqrt_normalisation:
                scaling = 1.0 / np.sqrt(n_motifs * level3_width)
            weights = self._get_weights(
                shape=[self.n_hidden, n_motifs, level3_width],
                name="tn5_layer_out",
                prior_mean=0.0,
                prior_sigma=scaling,
            )
            weights = (
                weights + weights.flip(-1)
            ) / torch.tensor(2.0, device=weights.device)
            # {N, in, L} & {out, in, L}
            x_rep = torch.nn.functional.conv1d(x_rhp, weights, padding="same")
            x_rep = x_rep / torch.tensor(0.2, device=x_rhp.device)

            # sum across positions
            # x_rep = torch.nn.functional.adaptive_avg_pool1d(x_rep, 1).squeeze(-1)
            x_re = x_rep.sum(dim=-1)

            # Alternative: use softplus to make the scores positive
            # and to have a smooth transition across detected values
            score = torch.nn.functional.softplus(
                x_re - torch.tensor(2.0, device=x_re.device)
            ).squeeze(-1) / torch.tensor(0.7, device=x_re.device)
        else:
            # sum across positions
            # x_rep = torch.nn.functional.adaptive_avg_pool1d(x_rep, 1).squeeze(-1)
            x_rh = x_rhp.sum(dim=-1)

            n_motifs = x_rh.shape[-1]
            ka_weight = self._get_positive_weights(
                shape=[1, n_motifs],
                name=f"tn5_layer_hill_ka",
                prior_alpha=2.0,
            )
            n_weights = self._get_weights(
                shape=[1, n_motifs],
                name="tn5_layer_hill_n",
                prior_mean=0.0,
                prior_sigma=1.0,
            )
            n_max = 4
            n_prior = 1
            n_weights = torch.sigmoid(
                n_weights / torch.tensor(20.0, device=n_weights.device)
                + torch.logit(torch.tensor(n_prior / n_max, device=n_weights.device))
            ) * torch.tensor(n_max, device=n_weights.device)
            x_rh = self.hill_function(x_rh, ka=ka_weight, n=n_weights)

            scaling = 1.0
            if self.use_sqrt_normalisation:
                scaling = 1.0 / np.sqrt(n_motifs)
            weights = self._get_weights(
                shape=[self.n_hidden, n_motifs],
                name="tn5_layer_out",
                prior_mean=0.0,
                prior_sigma=scaling,
            )
            scaling = 1.0
            if self.use_sqrt_normalisation:
                scaling = 1.0 / np.sqrt(n_motifs * n_motifs)
            weights_pairwise = self._get_weights(
                shape=[self.n_hidden, n_motifs, n_motifs],
                name="tn5_layer_out_pairwise",
                prior_mean=0.0,
                prior_sigma=scaling,
            )

            x_re = torch.einsum(
                "rh,zh->rz",
                x_rh,
                weights,
            ) + torch.einsum(
                "rh,zhm,rm->rz",
                x_rh,
                weights_pairwise,
                x_rh,
            )

            score = torch.nn.functional.softplus(
                x_re - torch.tensor(2.0, device=x_re.device)
            ).squeeze(-1) / torch.tensor(0.7, device=x_re.device)

        return score