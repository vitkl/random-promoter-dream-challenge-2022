from typing import Optional

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

    def __init__(
        self,
        name: str = "",
        fixed_motifs: Optional[np.array] = None,
        n_motifs: int = None,
        n_binding_modes: int = 4,
        use_motif_refining: bool = True,
        mode: str = "one_layer_conv2d",
        use_refined_motifs_below_zero: bool = False,
        motif_length: int = 5,
        seq_length: int = 7,
        n_nucleotides: int = 4,
        motif_loc_mean: float = 0.0,
        motif_loc_scale: float = 0.1,
        motif_weight_alpha: float = 10.0,
        n_hidden: int = 128,
        activation_fn: torch.nn.Module = Exp,
        pool_fn: torch.nn.Module = torch.nn.MaxPool2d,
        pool_window: Optional[int] = None,
        pool_stride: Optional[int] = None,
        use_reverse_complement: bool = False,
        return_forward_revcomp: bool = False,
        shift_seq_val: int = 0,
        use_affinity_scaling_by_motif_complexity: bool = True,
        use_affinity_scaling_by_motif_complexity_detach: bool = False,
        padding: str = "same",
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

        if self.use_motif_refining:
            # Motif weights
            weights_name = f"{name}motif_loc"
            if getattr(self.weights, weights_name, None) is None:
                deep_setattr(
                    self.weights,
                    weights_name,
                    PyroSample(
                        lambda prior: dist.SoftLaplace(
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
            self.register_buffer(f"{name}prior_alpha", torch.tensor(prior_alpha))
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
            self.register_buffer(f"{name}prior_mean", torch.tensor(prior_mean))
        if prior_sigma is None:
            prior_sigma = 1.0
        if getattr(self, f"{name}prior_sigma", None) is None:
            self.register_buffer(f"{name}prior_sigma", torch.tensor(prior_sigma))
        if getattr(self, f"{name}motif_length_tensor", None) is None:
            self.register_buffer(
                f"{name}motif_length_tensor", torch.tensor(self.motif_length)
            )
        # Motif weights
        weights_name = f"{name}laplace_weight"
        if getattr(self.weights, weights_name, None) is None:
            deep_setattr(
                self.weights,
                weights_name,
                PyroSample(
                    lambda prior: dist.SoftLaplace(
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
            # assert torch.allclose(
            #     torch.nn.functional.conv2d(x, motifs[:, :, :, self.complement_index].flip(-2)).squeeze(-1),
            #     torch.nn.functional.conv2d(x[:, :, :, self.complement_index].flip(-2), motifs).flip(-2).squeeze(-1)
            # )
            reverse_complement = torch.nn.functional.conv2d(
                x,
                motifs[:, :, :, self.complement_index].flip(-2),
                padding=padding,
            )
            if self.use_batch_norm:
                reverse_complement = self.batch_norm1(reverse_complement)
            elif self.use_layer_norm:
                reverse_complement = self.layer_norm1(reverse_complement)
        else:
            reverse_complement = None
        x = torch.nn.functional.conv2d(x, motifs)
        # output shape: (n_regions, n_motifs, seq_length - (motif_length - 1), 1)

        # Apply batch norm or layer norm
        if self.use_batch_norm:
            # TODO this needs to be 2d
            x = self.batch_norm1(x)
        elif self.use_layer_norm:
            # TODO this needs n_motifs as last dimension
            x = self.layer_norm1(x)
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
        # print("DNA sequence NN fraction", (x == torch.tensor(0.25, device=x.device)).float().mean())
        # get motif weights
        motifs = self._get_motif_weights(use_prior=True)

        # Do convolution
        if not self.use_einsum:
            x, reverse_complement = self._apply_conv2d(x, motifs, padding=self.padding)
        else:
            x, reverse_complement = self._apply_conv2d_einsum(
                x, motifs, equation="rpn,hwn->rhp"
            )

        # Apply batch norm or layer norm
        if self.use_batch_norm:
            # TODO this needs to be 2d
            x = self.batch_norm1(x)
        elif self.use_layer_norm:
            # TODO this needs n_motifs as last dimension
            x = self.layer_norm1(x)

        # Apply activation
        x = self.activation1(x)
        if self.use_reverse_complement and not self.return_forward_revcomp:
            x = x + self.activation1(reverse_complement)
        elif self.return_forward_revcomp:
            x = torch.cat([x, self.activation1(reverse_complement)], dim=-1)

        # apply pooling
        x = self.pool1(x)

        # print("m_binding.mean() TF", x.mean((-3, -1)))  # rhp
        # print("m_binding.max() TF", x.max(-3)[0].max(-1)[0])  # rhp
        # print("m_binding max - min TF", x.max(-3)[0].max(-1)[0] - x.min(-3)[0].min(-1)[0])  # rhp
        # print("m_binding max - min TF total", x.max(-3)[0].max(-1)[0].max() - x.max(-3)[0].max(-1)[0].min())  # rhp

        # print("m_binding.mean()", x.mean())
        # print("m_binding.max()", x.max())
        # print("m_binding.min()", x.min())

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
            # print("motif_complexity mean", motif_complexity.mean())
            x = torch.einsum("rhp,h->rhp", x, motif_complexity)
        else:
            x = x / torch.tensor(50.0, device=motifs.device)
        return x

    def two_layer_conv1d(
        self, x, region_motif_coo_rhp: torch.Tensor = None, width: int = 5
    ):
        # Layer 1 - learning Tn5-DNA motif  ===============================
        x_rhp = self.one_layer_conv2d(x)
        if region_motif_coo_rhp is not None:
            x_rhp = torch.concat(
                [
                    x_rhp,
                    region_motif_coo_rhp,
                ],
                dim=-2,
            )
        n_motifs = x_rhp.shape[-2]

        # Layer 2 - learning interactions between Tn5 sites ===============
        weights = self._get_weights(
            shape=[self.n_motifs, n_motifs, width],
            name="tn5_layer2",
            prior_mean=0.0,
            prior_sigma=1.0 / np.sqrt(n_motifs * width),
        )
        residual_weight = self._get_positive_weights(
            shape=[1, n_motifs, 1],
            name="tn5_layer2_residual",
            prior_alpha=2.0,
        )
        # {N, in, L} & {out, in, L} + residual
        if region_motif_coo_rhp is not None:
            x_rhp = torch.concat(
                [
                    torch.nn.functional.conv1d(x_rhp, weights, padding="same"),
                    x_rhp * residual_weight,
                ],
                dim=-2,
            )
        else:
            x_rhp = (
                torch.nn.functional.conv1d(x_rhp, weights, padding="same")
                + x_rhp * residual_weight
            )
        x_rhp = torch.nn.functional.softplus(x_rhp)

        n_motifs = x_rhp.shape[-2]
        # Layer 3 - summing up effects per batch per position ===============
        weights = self._get_weights(
            shape=[self.n_hidden, n_motifs, width],
            name="tn5_layer3",
            prior_mean=0.0,
            prior_sigma=1.0 / np.sqrt(n_motifs * width),
        )
        # {N, in, L} & {out, in, L}
        x_rep = torch.nn.functional.conv1d(x_rhp, weights, padding=self.padding)

        return x_rep