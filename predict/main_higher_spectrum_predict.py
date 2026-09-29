"""Model definition for spectrum completion with signed coefficients; no training loop."""

import math
import datetime
from typing import Any
import sys
import os
import csv
import multiprocessing
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.distributions import Bernoulli, Beta, Categorical, MixtureSameFamily, NegativeBinomial
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pyspark.sql.dataframe as ps

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import ROCmSafeLayerNorm


def build_data(
        sampler: TheorySampler,
        train_ratio: float,
        test_ratio: float,
        lower_cutoff: float,
        higher_cutoff: float,
        *,
        seed: int=42
):
    if not (
            math.isfinite(lower_cutoff)
            and math.isfinite(higher_cutoff)
            and 0 < lower_cutoff < higher_cutoff
    ):
        raise ValueError("Require finite cutoffs with 0 < lower < higher.")

    if not all(
            math.isfinite(r) and 0 <= r <= 1
            for r in (train_ratio, test_ratio)
    ):
        raise ValueError("Ratios must be finite and between 0 and 1.")

    ratio_sum = train_ratio + test_ratio
    if ratio_sum > 1 and not math.isclose(
            ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Training and test ratios must sum to at most 1.")

    valid_ratio = max(0.0, 1.0 - ratio_sum)

    train_set = [[], []]
    test_set = [[], []]
    validation_set = [[], []]

    train_df, test_df, valid_df = sampler.df.randomSplit([train_ratio, test_ratio, valid_ratio], seed=seed)

    def build_dataset(df: ps.DataFrame, dataset: list[Any]):
        def parse_partition(rows):
            for row in rows:
                lower_spec = []  # list of [dimension, coefficient]
                higher_spec = []

                sci = SuperConformalIndex(row["SCI"])
                for dim in sorted(sci.spectrum.keys()):
                    if dim <= lower_cutoff:
                        lower_spec.append([dim, sci.spectrum[dim]])
                    elif dim <= higher_cutoff:
                        higher_spec.append([dim, sci.spectrum[dim]])

                if lower_spec:
                    yield lower_spec, higher_spec

        pairs = (
            df.select("SCI")
            .repartition(multiprocessing.cpu_count())
            .rdd
            .mapPartitions(parse_partition)
            .collect()
        )

        for lower_spec, higher_spec in pairs:
            dataset[0].append(lower_spec)
            dataset[1].append(higher_spec)

    build_dataset(train_df, train_set)
    build_dataset(test_df, test_set)
    build_dataset(valid_df, validation_set)

    return train_set, test_set, validation_set


class SpectrumPredictDataset(Dataset):
    def __init__(self, x, y_true):
        if len(x) != len(y_true):
            raise ValueError("x and y_true must have the same number of theories.")

        self.x = x
        self.y_true = y_true

    def __getitem__(self, index):
        return self.x[index], self.y_true[index]

    def __len__(self):
        return len(self.x)

    @staticmethod
    def collate_spectra(batch):
        xs, ys = [], []

        for x_i, y_i in batch:
            x_i = torch.as_tensor(x_i, dtype=torch.float32)
            y_i = torch.as_tensor(y_i, dtype=torch.float32)

            if y_i.numel() == 0:
                y_i = y_i.reshape(0, 2)

            if (
                x_i.ndim != 2
                or x_i.shape[1] != 2
                or x_i.shape[0] == 0
            ):
                raise ValueError("Each input must be a nonempty [L, 2] array.")

            if y_i.ndim != 2 or y_i.shape[1] != 2:
                raise ValueError("Each target must be a [M, 2] array.")

            xs.append(x_i)
            ys.append(y_i)

        x_lengths = torch.tensor([len(x_i) for x_i in xs])
        y_lengths = torch.tensor([len(y_i) for y_i in ys])

        x = pad_sequence(xs, batch_first=True, padding_value=0.0)
        y_true = pad_sequence(ys, batch_first=True, padding_value=0.0)

        src_pad = torch.arange(x.size(1))[None, :] >= x_lengths[:, None]
        tgt_pad = torch.arange(y_true.size(1))[None, :] >= y_lengths[:, None]

        return x, y_true, src_pad, tgt_pad


class HigherSpectrumPredictModel(nn.Module):
    """Set-attention encoder and autoregressive higher-spectrum decoder.

    Inputs contain raw ``(dimension, coefficient)`` pairs, not precomputed
    features. ``x`` has shape [B, L, 2], with dimensions <= input_cutoff.
    ``previous_y`` has shape [B, M, 2], with strictly increasing dimensions in
    (input_cutoff, output_cutoff]. Real coefficients must be nonzero signed
    integers; omit zero-coefficient levels. The pair layout and dimension
    cutoffs are unchanged. Positive input coefficients are still supported.
    Inputs must use the model's floating dtype and device.

    ``src_pad`` and ``tgt_pad`` are Boolean masks of shapes [B, L] and [B, M];
    True means padding. Omitted masks mean every pair is real. Source order is
    arbitrary, but target padding must follow all real target pairs. Every
    source spectrum must contain at least one real level.

    The model prepends BOS itself and returns M + 1 prediction steps. For
    teacher forcing, pass the complete y_true as previous_y: step j predicts
    y_true[:, j] using only earlier pairs, and the step after the last real
    pair predicts STOP. Only STOP is supervised at that terminal step; steps
    marked by the returned step_pad must be ignored altogether.

    Returned tensors:
        hidden: [B, M + 1, d_model], for conditioning the coefficient head.
        stop_logits: [B, M + 1], log-odds of ending the spectrum.
        endpoint_logits: [B, M + 1], log-odds of the next dimension being
            exactly output_cutoff, conditional on continuing.
        gap_logits, gap_alpha, gap_beta: [B, M + 1, num_gap_components],
            parameters for the fractional gap u in (0, 1), conditional on
            continuing and not choosing the exact endpoint.
        previous_dimensions: [B, M + 1], starting at input_cutoff.
        step_pad: [B, M + 1], including an unmasked BOS prediction step.

    If next_dimensions [B, M + 1] is supplied, also return:
        count_sign_logits: [B, M + 1], log-odds of a positive coefficient.
        count_positive_total_count, count_positive_logits: [B, M + 1],
            NegativeBinomial parameters for abs(coefficient)-1 given c > 0.
        count_negative_total_count, count_negative_logits: [B, M + 1],
            NegativeBinomial parameters for abs(coefficient)-1 given c < 0.
    Supply true next dimensions during training and selected next dimensions
    during inference. Finite placeholders are allowed at STOP and padding
    steps, where coefficient outputs must be ignored. These dimensions
    condition only the coefficient head, never the decoder or other heads.

    For inference, encode once with encode(), then repeatedly call decode()
    on the generated prefix. Decode with previous_y=None to predict the first
    level from BOS alone. Select STOP first, otherwise select a dimension,
    then use multiplicity_distribution() to select its sign and magnitude.
    This method returns sign and conditional magnitude distributions, not a
    single NegativeBinomial distribution. Selecting the
    exact output_cutoff endpoint necessarily ends the spectrum afterward.
    Generation policy, losses, data loading, and optimization are not included.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_gap_components: int = 5,
        input_cutoff: float = 2.25,
        output_cutoff: float = 4.5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        sizes = (d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, num_gap_components)
        if any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError("Model sizes and layer counts must be positive integers.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1].")
        if not (math.isfinite(input_cutoff) and math.isfinite(output_cutoff)
                and 0 < input_cutoff < output_cutoff):
            raise ValueError("Cutoffs must be finite and satisfy 0 < input < output.")

        self.d_model = d_model
        self.num_gap_components = num_gap_components
        self.input_cutoff = float(input_cutoff)
        self.output_cutoff = float(output_cutoff)

        self.src_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.tgt_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.bos = nn.Parameter(torch.empty(1, 1, d_model))

        layer_options = dict(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**layer_options),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.stop_head = nn.Linear(d_model, 1)
        self.endpoint_head = nn.Linear(d_model, 1)
        self.gap_head = nn.Linear(d_model, 3 * num_gap_components)
        self.count_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 5),
        )

        # Transformer layers construct their own LayerNorms. Replace those as
        # well as the final norms, reusing parameters for checkpoint compatibility.
        for module in tuple(self.modules()):
            for name, child in tuple(module.named_children()):
                if isinstance(child, nn.LayerNorm):
                    replacement = ROCmSafeLayerNorm(
                        child.normalized_shape, eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                        bias=child.bias is not None,
                    )
                    replacement.weight = child.weight
                    replacement.bias = child.bias
                    setattr(module, name, replacement)

        # PyTorch clones the prototype layers; initialize each clone separately.
        self.apply(self._initialize)
        nn.init.normal_(self.bos, std=0.02)

        self.to(dtype=dtype)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    @staticmethod
    def _padding_mask(
        mask: torch.Tensor | None, shape: tuple[int, int], device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros(shape, dtype=torch.bool, device=device)
        if mask.dtype != torch.bool or tuple(mask.shape) != shape:
            raise ValueError(f"{name} must be a Boolean tensor with shape {shape}.")
        if mask.device != device:
            raise ValueError(f"{name} must be on the same device as its input.")
        return mask

    def _prepare_pairs(
        self, pairs: torch.Tensor, padding: torch.Tensor | None, *, target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        name = "previous_y" if target else "x"
        if pairs.ndim != 3 or pairs.size(-1) != 2 or pairs.size(0) == 0:
            raise ValueError(f"{name} must have shape [B, length, 2] with B > 0.")
        if pairs.dtype != self.bos.dtype or pairs.device != self.bos.device:
            raise ValueError(f"{name} must have the model's floating dtype and device.")
        padding = self._padding_mask(
            padding, tuple(pairs.shape[:2]), pairs.device,
            "tgt_pad" if target else "src_pad",
        )
        # Sanitize masked values before log1p; padding never supplies a feature.
        pairs = pairs.masked_fill(padding.unsqueeze(-1), 0)
        real = pairs[~padding]
        if not torch.isfinite(real).all():
            raise ValueError(f"Real {name} pairs must be finite.")
        if torch.any((real[:, 1] == 0) | (real[:, 1] != real[:, 1].round())):
            raise ValueError("Real coefficients must be nonzero integers.")
        dimensions = real[:, 0]
        if target:
            if torch.any(padding[:, :-1] & ~padding[:, 1:]):
                raise ValueError("tgt_pad must describe right-padded target spectra.")
            if torch.any((dimensions <= self.input_cutoff)
                         | (dimensions > self.output_cutoff)):
                raise ValueError("Target dimensions must be in (input_cutoff, output_cutoff].")
            differences = pairs[:, 1:, 0] - pairs[:, :-1, 0]
            if torch.any((differences <= 0) & ~padding[:, 1:]):
                raise ValueError("Real target dimensions must be strictly increasing.")
        else:
            if torch.any(padding.all(dim=1)):
                raise ValueError("Each source spectrum needs at least one real level.")
            if torch.any((dimensions < 0) | (dimensions > self.input_cutoff)):
                raise ValueError("Source dimensions must be in [0, input_cutoff].")
        return pairs, padding

    @staticmethod
    def _features(pairs: torch.Tensor, cutoff: float) -> torch.Tensor:
        coefficient = pairs[..., 1]
        signed_log = coefficient.sign() * torch.log1p(coefficient.abs())
        return torch.stack((pairs[..., 0] / cutoff, signed_log), dim=-1)

    def _positions(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        # Dynamic sinusoidal positions avoid a fixed maximum target length.
        dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
        positions = torch.arange(length, device=reference.device, dtype=dtype)[:, None]
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=reference.device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions * frequencies
        encoding = torch.zeros(length, self.d_model, device=reference.device, dtype=dtype)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :self.d_model // 2].cos()
        return encoding.to(dtype=reference.dtype).unsqueeze(0)

    def encode(self, x: torch.Tensor, src_pad: torch.Tensor | None = None) -> torch.Tensor:
        """Return [B, L, d_model] memory without source positional encodings."""
        x, src_pad = self._prepare_pairs(x, src_pad, target=False)
        memory = self.encoder(
            self.src_embed(self._features(x, self.input_cutoff)),
            src_key_padding_mask=src_pad,
        )
        return memory.masked_fill(src_pad.unsqueeze(-1), 0)

    def decode(
        self,
        memory: torch.Tensor,
        previous_y: torch.Tensor | None = None,
        src_pad: torch.Tensor | None = None,
        tgt_pad: torch.Tensor | None = None,
        next_dimensions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict after each prefix, including BOS; reuse encode() memory.

        Pass the same src_pad used by encode(). tgt_pad covers previous_y
        only, not BOS. With no previous_y, return a single prediction step.
        """
        if memory.ndim != 3 or memory.size(-1) != self.d_model or memory.size(0) == 0:
            raise ValueError("memory must have shape [B, L, d_model] with B > 0.")
        batch_size = memory.size(0)
        src_pad = self._padding_mask(
            src_pad, tuple(memory.shape[:2]), memory.device, "src_pad",
        )
        if torch.any(src_pad.all(dim=1)):
            raise ValueError("Each source spectrum needs at least one real level.")
        if previous_y is None:
            previous_y = self.bos.new_empty(batch_size, 0, 2)
        previous_y, tgt_pad = self._prepare_pairs(previous_y, tgt_pad, target=True)
        if previous_y.size(0) != batch_size:
            raise ValueError("Source and target batch sizes must match.")

        embedded = self.tgt_embed(self._features(previous_y, self.output_cutoff))
        embedded = torch.cat((self.bos.expand(batch_size, -1, -1), embedded), dim=1)
        embedded = embedded + self._positions(embedded.size(1), embedded)
        step_pad = torch.cat((tgt_pad.new_zeros(batch_size, 1), tgt_pad), dim=1)
        causal_mask = torch.ones(
            embedded.size(1), embedded.size(1), device=embedded.device, dtype=torch.bool,
        ).triu(diagonal=1)
        hidden = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=step_pad,
            memory_key_padding_mask=src_pad,
        ).masked_fill(step_pad.unsqueeze(-1), 0)

        gap_logits, raw_alpha, raw_beta = self.gap_head(hidden).chunk(3, dim=-1)
        prediction = {
            "hidden": hidden,
            "stop_logits": self.stop_head(hidden).squeeze(-1),
            "endpoint_logits": self.endpoint_head(hidden).squeeze(-1),
            "gap_logits": gap_logits,
            "gap_alpha": F.softplus(raw_alpha) + 1e-4,
            "gap_beta": F.softplus(raw_beta) + 1e-4,
            "previous_dimensions": torch.cat((
                previous_y.new_full((batch_size, 1), self.input_cutoff),
                previous_y[..., 0],
            ), dim=1),
            "step_pad": step_pad,
        }
        if next_dimensions is not None:
            prediction.update(self.multiplicity_parameters(hidden, next_dimensions))
        return prediction

    def forward(
        self,
        x: torch.Tensor,
        previous_y: torch.Tensor | None = None,
        src_pad: torch.Tensor | None = None,
        tgt_pad: torch.Tensor | None = None,
        next_dimensions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode raw observed pairs and decode the supplied higher-level prefix."""
        memory = self.encode(x, src_pad)
        return self.decode(memory, previous_y, src_pad, tgt_pad, next_dimensions)

    @staticmethod
    def gap_distribution(prediction: dict[str, torch.Tensor]) -> MixtureSameFamily:
        """Distribution of u=(next-previous)/(output_cutoff-previous).

        Conditional on continuing and not selecting the endpoint. For a
        density in dimension units, divide the u density by the remaining
        interval length. STOP, endpoint, and padding steps have no Beta loss.
        """
        return MixtureSameFamily(
            Categorical(logits=prediction["gap_logits"]),
            Beta(prediction["gap_alpha"], prediction["gap_beta"]),
        )

    def dimension_from_gap(
        self, previous_dimensions: torch.Tensor, fractional_gap: torch.Tensor,
    ) -> torch.Tensor:
        """Map 0 < u <= 1 to the remaining interval; u=1 selects the endpoint.

        Only call for continuation steps with previous dimension < cutoff.
        In finite precision, reject a sampled result that rounds back to the
        previous dimension instead of appending a duplicate level.
        """
        return previous_dimensions + (self.output_cutoff - previous_dimensions) * fractional_gap

    def multiplicity_parameters(
        self, hidden: torch.Tensor, next_dimensions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Condition the sign and magnitude heads on a true or selected dimension.

        next_dimensions must match hidden.shape[:-1]. Single-step hidden
        vectors [B, d_model] and dimensions [B] are also supported.
        """
        if next_dimensions.shape != hidden.shape[:-1]:
            raise ValueError("next_dimensions must have shape hidden.shape[:-1].")
        features = torch.cat((
            hidden, (next_dimensions / self.output_cutoff).unsqueeze(-1),
        ), dim=-1)
        sign_logits, raw_positive_count, positive_logits, raw_negative_count, negative_logits = (
            self.count_head(features).unbind(dim=-1)
        )
        return {
            "count_sign_logits": sign_logits,
            "count_positive_total_count": F.softplus(raw_positive_count) + 1e-4,
            "count_positive_logits": positive_logits,
            "count_negative_total_count": F.softplus(raw_negative_count) + 1e-4,
            "count_negative_logits": negative_logits,
        }

    def multiplicity_distribution(
        self, hidden: torch.Tensor, next_dimensions: torch.Tensor,
    ) -> dict[str, Bernoulli | NegativeBinomial]:
        """Return sign and conditional abs(coefficient)-1 distributions.

        ``sign`` is Bernoulli: 1 means positive, 0 means negative.
        ``positive`` and ``negative`` are NegativeBinomial distributions of
        k = abs(c)-1 conditional on the corresponding sign. Sample sign first,
        then sample k from its branch and form c = (2*sign-1)*(k+1).

        For a nonzero true coefficient c, its log probability is
        sign.log_prob((c > 0).float()) plus the selected branch's
        log_prob(abs(c)-1). Ignore STOP and padding positions. Zero has no
        probability in this sparse representation.
        """
        parameters = self.multiplicity_parameters(hidden, next_dimensions)
        return {
            "sign": Bernoulli(logits=parameters["count_sign_logits"]),
            "positive": NegativeBinomial(
                total_count=parameters["count_positive_total_count"],
                logits=parameters["count_positive_logits"],
            ),
            "negative": NegativeBinomial(
                total_count=parameters["count_negative_total_count"],
                logits=parameters["count_negative_logits"],
            ),
        }


def spectrum_loss(
        model: HigherSpectrumPredictModel,
        x: torch.Tensor,
        y_true: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_pad: torch.Tensor,
):
    """
    Compute spectrum prediction loss.
    @param model: HigherSpectrumPredictModel
    @param x: input tensor
    @param y_true: real output tensor
    @param src_pad: padding tensor for x. False if real value, True if empty value.
    @param tgt_pad: padding tensor for y_true.
    """
    out = model(
        x,
        previous_y=y_true,
        src_pad=src_pad,
        tgt_pad=tgt_pad,
    )

    B, M, _ = y_true.shape
    lengths = (~tgt_pad).sum(dim=1)[:, None]
    steps = torch.arange(M + 1, device=y_true.device)[None, :]

    pair_valid = steps < lengths
    stop_valid = steps <= lengths
    stop_true = (steps == lengths).to(y_true.dtype)

    targets = torch.cat(
        (y_true, y_true.new_zeros(B, 1, 2)),
        dim=1
    )
    dims, coeffs = targets.unbind(dim=-1)

    endpoint = pair_valid & (dims == model.output_cutoff)
    interior = pair_valid & ~endpoint

    zero = out["hidden"].new_zeros(())
    parts = {
        "endpoint": zero,
        "gap": zero,
        "sign": zero,
        "magnitude": zero
    }

    parts["stop"] = F.binary_cross_entropy_with_logits(
        out["stop_logits"][stop_valid],
        stop_true[stop_valid],
        reduction="sum"
    )

    if pair_valid.any():
        parts["endpoint"] = F.binary_cross_entropy_with_logits(
            out["endpoint_logits"][pair_valid],
            endpoint[pair_valid].to(y_true.dtype),
            reduction="sum"
        )

        distributions = model.multiplicity_distribution(
            out["hidden"][pair_valid],
            dims[pair_valid],
        )

        c = coeffs[pair_valid]
        positive = c > 0
        k = c.abs() - 1

        parts["sign"] = -distributions["sign"].log_prob(positive.to(c.dtype)).sum()
        parts["magnitude"] = -torch.where(
            positive,
            distributions["positive"].log_prob(k),
            distributions["negative"].log_prob(k),
        ).sum()

    if interior.any():
        previous = out["previous_dimensions"][interior]
        remaining = model.output_cutoff - previous
        u = (dims[interior] - previous) / remaining

        gap_parameters = {
            name: out[name][interior]
            for name in ("gap_logits", "gap_alpha", "gap_beta")
        }
        gap_dist = model.gap_distribution(gap_parameters)

        # Convert density in fractional-gap units to dimension units.
        parts["gap"] = (
            -gap_dist.log_prob(u) + remaining.log()
        ).sum()

    parts = {name: value / B for name, value in parts.items()}
    return sum(parts.values()), parts


@torch.inference_mode()
def predict_batch(
        model: HigherSpectrumPredictModel,
        x: torch.Tensor,
        src_pad: torch.Tensor | None = None,
        max_levels: int=512,
        sample: bool=False,
):
    """
    Predict higher spectrum from lower spectrum. Call eval for model before calling
    this function.
    @param model: HigherSpectrumPredictModel
    @param x: input tensor
    @param src_pad: padding tensor for x. False if real value, True if empty value.
    @param max_levels: maximum number of levels to predict.
    @param sample: whether to sample or not. If sampled, generate different
    prediction from same input using probability distribution.
    """
    if x.ndim != 3 or x.size(0) == 0 or max_levels <= 0:
        raise ValueError(
            "Use a nonempty batch and a positive max levels."
        )

    if src_pad is None:
        src_pad = torch.zeros(
            x.shape[:2], dtype=torch.bool, device=x.device
        )

    def choose(logits):
        if sample:
            return torch.distributions.Bernoulli(
                logits=logits
            ).sample().bool()
        return logits >= 0 # probability is larger than 0.5 if logits >= 0.

    B = x.size(0)
    memory = model.encode(x, src_pad)

    predicted = x.new_zeros(B, max_levels, 2)
    lengths = torch.zeros(B, dtype=torch.long, device=x.device)
    reasons = [None] * B
    active = torch.arange(B, device=x.device)

    def finish(ids, reason):
        for id in ids.tolist():
            reasons[id] = reason

    for step in range(max_levels + 1):
        if active.numel() == 0:
            break

        out = model.decode(
            memory[active],
            previous_y = predicted[active, :step],
            src_pad=src_pad[active],
        )
        last = {
            name: value[:, -1]
            for name, value in out.items()
        }

        stop = choose(last["stop_logits"])
        finish(active[stop], "stop")
        active = active[~stop]
        last = {
            name: value[~stop]
            for name, value in last.items()
        }

        if active.numel() == 0:
            break

        if step == max_levels:
            finish(active, "max_levels")
            break

        endpoint = choose(last["endpoint_logits"])
        dims = x.new_full(
            (active.numel(),),
            model.output_cutoff,
        )
        interior = ~endpoint

        if interior.any():
            gap = {
                name: last[name][interior]
                for name in ("gap_logits", "gap_alpha", "gap_beta")
            }

            if sample:
                u = model.gap_distribution(gap).sample()
            else:
                component = gap["gap_logits"].argmax(-1, keepdim=True)
                alpha = gap["gap_alpha"].gather(-1, component).squeeze(-1)
                beta = gap["gap_beta"].gather(-1, component).squeeze(-1)
                u = alpha / (alpha + beta) # Mean of beta distribution

            dims[interior] = model.dimension_from_gap(last["previous_dimensions"][interior], u)

        valid = (
            torch.isfinite(dims)
            & (dims > last["previous_dimensions"])
            & (endpoint | (dims < model.output_cutoff))
        )
        finish(active[~valid], "numerical_limit")

        active = active[valid]
        dims = dims[valid]
        endpoint = endpoint[valid]
        hidden = last["hidden"][valid]

        if active.numel() == 0:
            break

        counts = model.multiplicity_distribution(hidden, dims)
        positive = choose(counts["sign"].logits)

        kp = counts["positive"].sample() if sample else counts["positive"].mode
        kn = counts["negative"].sample() if sample else counts["negative"].mode
        coefficients = torch.where(positive, kp + 1, -(kn + 1))

        valid = torch.isfinite(coefficients)
        finish(active[~valid], "numerical_limit")

        active = active[valid]
        dims = dims[valid]
        endpoint = endpoint[valid]
        coefficients = coefficients[valid]

        predicted[active, step] = torch.stack((dims, coefficients), dim=-1)
        lengths[active] = step + 1

        finish(active[endpoint], "upper_cutoff")
        active = active[~endpoint]

    max_length = int(lengths.max().item())
    return predicted[:, :max_length], lengths, reasons


def spectrum_error(y_true, y_pred, tolerance=1e-3):
    """
    Calculate the error and accuracy of the spectrum expectation model.
    :param y_true: true higher spectrum
    :param y_pred: predicted higher spectrum
    :param tolerance: tolerance to identify the dimension. Default valud is 1e-3.
    :return: statistics of the expectation.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    def prepare_spectrum(values):
        array = np.asarray(values, dtype=np.float64)

        if array.shape == (0,):
            array = array.reshape(0, 2)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Pass one unpadded spectrum with shape [N, 2]")
        if not np.isfinite(array).all():
            raise ValueError("Spectrum values must be finite")
        if np.any(array[:, 1] == 0):
            raise ValueError("Remove padding and zero-coefficient levels.")

        return array

    true = prepare_spectrum(y_true)
    pred = prepare_spectrum(y_pred)

    nt, npred = len(true), len(pred)
    if nt and npred:
        distances = np.abs(true[:, None, 0] - pred[None, :, 0])

        forbidden_cost = min(nt, npred) + 1.0
        cost = np.where(distances <= tolerance, distances / tolerance, forbidden_cost)

        rows, cols = linear_sum_assignment(cost)

        allowed = distances[rows, cols] <= tolerance
        rows, cols = rows[allowed], cols[allowed]
    else:
        rows = cols = np.empty(0, dtype=int)

    matched = len(rows)
    extra = npred - matched
    missing = nt - matched

    precision = matched / npred if npred else float(nt == 0)
    recall = matched / nt if nt else 1.0
    f1 = 2 * matched / (nt + npred) if nt + npred else 1.0

    if matched:
        dimension_mae = np.abs(true[rows, 0] - pred[cols, 0]).mean()
        coefficient_mae = np.abs(true[rows, 1] - pred[cols, 1]).mean()
        coefficient_acc = np.mean(true[rows, 1] == pred[cols, 1])
        sign_acc = np.mean(np.sign(true[rows, 1]) == np.sign(pred[cols, 1]))
    else:
        dimension_mae = None
        coefficient_mae = None
        coefficient_acc = None
        sign_acc = None

    return {
        "matched": matched,
        "extra": extra,
        "missing": missing,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dimension_mae": dimension_mae,
        "coefficient_mae": coefficient_mae,
        "coefficient_acc": coefficient_acc,
        "sign_acc": sign_acc,
    }


@torch.no_grad()
def _clip_training_gradients(model: nn.Module, max_norm: float = 1.0) -> torch.Tensor:
    """Keep the normal fast path, with a float64 fallback for norm overflow."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    try:
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=max_norm, error_if_nonfinite=True,
        )
    except RuntimeError as error:
        # clip_grad_norm_ checks the norm before changing any gradients.
        if "is non-finite" not in str(error):
            raise

        # Inspect on CPU as well, to distinguish bad entries from a GPU
        # reduction problem. This slower path runs only when clipping fails.
        gradients = [
            (name, p.grad, p.grad.detach().to(device="cpu", dtype=torch.float64))
            for name, p in model.named_parameters() if p.grad is not None
        ]
        bad = [
            f"{name}: NaN={int(g.isnan().sum())}, Inf={int(g.isinf().sum())}"
            for name, _, g in gradients if not torch.isfinite(g).all()
        ]
        if bad:
            raise FloatingPointError(
                "Non-finite gradient entries; optimizer update cancelled. "
                + "; ".join(bad)
            ) from error

        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g) for _, _, g in gradients
        ]))
        if not torch.isfinite(norm):
            raise FloatingPointError(
                "Gradient norm is non-finite even in float64; optimizer update cancelled."
            ) from error

        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for _, destination, g in gradients:
            # Scale before converting back, avoiding an underflowing float32
            # scale factor when the original finite gradients are very large.
            destination.copy_(g * scale)
        warnings.warn(
            f"Gradient entries were finite, but the original norm calculation "
            f"failed. Clipped using CPU float64 norm {norm.item():.6g}.",
            RuntimeWarning, stacklevel=2,
        )
        return norm


def _save_gradient_failure(model, batch, parts, batch_index, c, rng_state) -> Path:
    """Save one backward-pass reproducer, separate from training checkpoints."""
    directory = Path(__file__).resolve().parents[1] / "data" / "predict" / "debug"
    directory.mkdir(parents=True, exist_ok=True)
    encoder_layer = model.encoder.layers[0]
    state = {
        "model_config": {
            "d_model": model.d_model,
            "nhead": encoder_layer.self_attn.num_heads,
            "num_encoder_layers": len(model.encoder.layers),
            "num_decoder_layers": len(model.decoder.layers),
            "dim_feedforward": encoder_layer.linear1.out_features,
            "dropout": encoder_layer.dropout.p,
            "num_gap_components": model.num_gap_components,
            "input_cutoff": model.input_cutoff,
            "output_cutoff": model.output_cutoff,
            "dtype": model.bos.dtype,
        },
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "gradients": {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None},
        "batch": tuple(t.detach().cpu() for t in batch),
        "loss_parts": {k: v.detach().cpu() for k, v in parts.items()},
        "batch_index": batch_index,
        "l1_coefficient": c,
        "rng_state": rng_state,
        "torch_version": str(torch.__version__),
        "device": str(model.bos.device),
    }
    with tempfile.NamedTemporaryFile(
        prefix="nonfinite-gradient-", suffix=".pt", dir=directory, delete=False,
    ) as snapshot:
        torch.save(state, snapshot)
        return Path(snapshot.name)


def train(loader: DataLoader, model: HigherSpectrumPredictModel, optimizer: Optimizer, device: torch.device, c: float=0.001):
    model.train()

    i = 1
    for x, y_true, src_pad, tgt_pad in loader:
        print(f"Train batch {i} training...")
        x = x.to(device)
        y_true = y_true.to(device)
        src_pad = src_pad.to(device)
        tgt_pad = tgt_pad.to(device)

        optimizer.zero_grad(set_to_none=True)
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(x.device) if x.is_cuda else None,
        }

        loss, parts = spectrum_loss(model, x, y_true, src_pad, tgt_pad)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += c * l1_norm

        if not torch.isfinite(loss).item():
            values = {
                name: value.detach().item()
                for name, value in parts.items()
            }
            raise FloatingPointError(f"Non-finite loss in batch {i}: {values}")

        loss.backward()

        try:
            _clip_training_gradients(model, max_norm=1.0)
        except FloatingPointError as error:
            path = _save_gradient_failure(
                model, (x, y_true, src_pad, tgt_pad), parts, i, c, rng_state,
            )
            raise FloatingPointError(f"{error}\nReproducer saved to {path}") from error

        optimizer.step()
        i += 1


def test(loader: DataLoader, model: HigherSpectrumPredictModel, device: torch.device, *, generate_prediction: bool = False):
    model.eval()

    results = []

    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for x, y_true, src_pad, tgt_pad in loader:
            x = x.to(device)
            y_true = y_true.to(device)
            src_pad = src_pad.to(device)
            tgt_pad = tgt_pad.to(device)

            test_loss, _ = spectrum_loss(model, x, y_true, src_pad, tgt_pad)
            total_loss += test_loss * x.size(0)
            total_num += x.size(0)

            if generate_prediction:
                y_pred, pred_lengths, reasons = predict_batch(model, x, src_pad=src_pad, sample=False)

                for i in range(x.size(0)):
                    true_yi = y_true[i, ~tgt_pad[i]].detach().cpu().numpy()
                    pred_yi = y_pred[i, :pred_lengths[i].item()].detach().cpu().numpy()

                    metrics = spectrum_error(true_yi, pred_yi)
                    metrics["termination"] = reasons[i]
                    results.append(metrics)
            else:
                results.append({})

    if not results:
        raise ValueError("The test loader contains no theories.")

    total: dict[str, int | float | None] = {
        "num_theories": len(results),
        "loss": total_loss / total_num if total_num else 0.0,
    }

    if generate_prediction:
        matched = sum(metric["matched"] for metric in results)
        extra = sum(metric["extra"] for metric in results)
        missing = sum(metric["missing"] for metric in results)

        n_true = matched + missing
        n_pred = matched + extra

        total |= {
            "matched": matched,
            "extra": extra,
            "missing": missing,
            "precision": matched / n_pred if n_pred else float(n_true == 0),
            "recall": matched / n_true if n_true else 1.0,
            "f1": 2 * matched / (n_true + n_pred) if n_true + n_pred else 1.0,
            "loss": total_loss / total_num if total_num else 0.0,
        }
        for key in ("dimension_mae", "coefficient_mae", "coefficient_acc", "sign_acc"):
            total[key] = sum(metric["matched"] * metric[key] for metric in results if metric[key] is not None) / matched if matched > 0 else None

    return total

if __name__ == "__main__":
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/predict', exist_ok=True)
    csv.field_size_limit(np.iinfo(np.int32).max)

    filename = input("Enter file name to load: ")

    theory_sampler = TheorySampler(filename)
    stats = theory_sampler.get_gauge_group_stats()
    for row in stats.collect():
        print(row.asDict())

    gauge_group = input("Enter gauge group to learn: ")
    learn_sampler = theory_sampler.get_selected_gauge_groups([gauge_group])

    lower_cutoff = float(input("Enter lower cutoff of the operator dimension: "))
    higher_cutoff = float(input("Enter higher cutoff of the operator dimension: "))
    train_ratio = float(input("Enter train ratio: "))
    test_ratio = float(input("Enter test ratio: "))

    train_set, test_set, validation_set = build_data(
        learn_sampler, train_ratio, test_ratio, lower_cutoff, higher_cutoff
    )

    dataset_train = SpectrumPredictDataset(train_set[0], train_set[1])
    dataset_test = SpectrumPredictDataset(test_set[0], test_set[1])
    dataset_validation = SpectrumPredictDataset(validation_set[0], validation_set[1])

    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, collate_fn=SpectrumPredictDataset.collate_spectra)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, collate_fn=SpectrumPredictDataset.collate_spectra)
    dataloader_validation = DataLoader(dataset_validation, batch_size=64, shuffle=False, collate_fn=SpectrumPredictDataset.collate_spectra)

    for index, (x, y_true, src_pad, tgt_pad) in enumerate(dataloader_train):
        print(f"{index + 1}/{len(dataloader_train)}", end=" ")
        print('x shape: ', x.shape, end=' ')
        print('y_true shape: ', y_true.shape, end=' ')
        print('src_pad shape: ', src_pad.shape, end=' ')
        print('tgt_pad shape: ', tgt_pad.shape)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = HigherSpectrumPredictModel(input_cutoff=lower_cutoff, output_cutoff=higher_cutoff).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 1e10

    n_epochs = int(input("Enter number of epochs: "))
    checkpoint_path = input("Enter the name of the checkpoint file: ")

    file_suffix = f"{gauge_group}_{lower_cutoff}_{higher_cutoff}"
    if os.path.isfile(checkpoint_path):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(n_epochs):
        print(f"Train epoch {epoch + 1}...")
        train(dataloader_train, model, optimizer, device)

        print(f"Test epoch {epoch + 1}...")
        total = test(dataloader_test, model, device, generate_prediction=(epoch + 1) % 5 == 0)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        for k, v in total.items():
            print(f"{k}: {v}")
        print()

        if total["loss"] < best_loss:
            best_loss = total["loss"]
            print("New best loss obtained. Saving model...")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

    print("Validating best model...")
    total = test(dataloader_validation, model, device, generate_prediction=True)
    for k, v in total.items():
        print(f"{k}: {v}")

    save_dir = f"../data/predict/{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/higher_spectrum_predict_{file_suffix}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for key in sorted(total.keys()):
            writer.writerow([key, str(total[key])])
