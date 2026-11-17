"""Parameter-matched LL segment refresh for the signed-Detail FSC SSM.

This is a separate experiment: the archived single-initial-state FSC v1 is
unchanged. All three modes have identical trainable parameters.
"""

import torch
from torch import nn
from torch.nn import functional as F

from e2_fsc_mamba import CONTEXT_SCALE, FSCMambaE2, LLStateTransfer


MODES = ("local_feature", "local_refresh", "local_shuffled")
VARIANT_PREFIX = "lwpd_e2_fsc_local_refresh_v1_"
CHUNK_SIZE = 16
CARRY = 0.5


def scan_with_refresh(scan, sequence, initial_state, refresh_states=None,
                      chunk_size=CHUNK_SIZE, carry=CARRY):
    """Selective SSM with fixed, state-space mixing at chunk boundaries.

    refresh_states: [batch, ceil(length/chunk_size), inner, state_channels].
    t=0 uses initial_state directly; later chunk starts use
    h <- carry*h + (1-carry)*h_LL, without resetting accumulated history.
    """
    if sequence.ndim != 3 or sequence.shape[-1] != scan.channels:
        raise ValueError("Expected BxLxC sequence")
    if initial_state.shape != (sequence.shape[0], scan.inner, scan.state_channels):
        raise ValueError("Incorrect initial state shape")
    if not isinstance(chunk_size, int) or chunk_size < 1 or not 0 <= carry <= 1:
        raise ValueError("Invalid chunk_size/carry")
    if refresh_states is not None:
        count = (sequence.shape[1] + chunk_size - 1) // chunk_size
        if refresh_states.shape != (sequence.shape[0], count, scan.inner, scan.state_channels):
            raise ValueError("Incorrect refresh_states shape")

    value, gate = scan.input_projection(scan.norm(sequence)).chunk(2, dim=-1)
    delta, b, c = scan.parameter_projection(value).split(
        (scan.inner, scan.state_channels, scan.state_channels), dim=-1,
    )
    delta = F.softplus(delta + scan.delta_bias).clamp(max=1.0)
    transition = -scan.a_log.exp()
    state = initial_state.to(transition.dtype)
    outputs = []
    for t in range(sequence.shape[1]):
        if refresh_states is not None and t > 0 and t % chunk_size == 0:
            state = carry * state + (1 - carry) * refresh_states[:, t // chunk_size].to(state.dtype)
        dt = delta[:, t, :, None]
        decay = torch.exp(dt * transition[None])
        drive = dt * b[:, t, None, :] * value[:, t, :, None]
        state = decay * state + drive
        outputs.append((state * c[:, t, None, :]).sum(-1) + scan.skip * value[:, t])
    return scan.output_projection(torch.stack(outputs, dim=1) * F.silu(gate))


class LocalRefreshTransfer(LLStateTransfer):
    """Refresh (or feature-inject) from matched LL chunks in both scan axes."""

    def __init__(self, previous, mode, chunk_size=CHUNK_SIZE, carry=CARRY):
        if mode not in MODES:
            raise ValueError(f"Unknown local FSC mode: {mode}")
        if not isinstance(chunk_size, int) or chunk_size < 1 or not 0 <= carry <= 1:
            raise ValueError("Invalid chunk size / carry")
        nn.Module.__init__(self)
        # Reuse the exact layers and weights initialized by FSC v1.
        self.core = previous.core
        self.ll_projection = previous.ll_projection
        self.decay_logit = previous.decay_logit
        self.mode = mode
        self.chunk_size = chunk_size
        self.carry = float(carry)

    def summaries(self, low_sequence):
        """Each chunk uses only the corresponding LL samples, not future chunks."""
        length = low_sequence.shape[1]
        if length < 1:
            raise ValueError("Empty scan")
        return torch.stack([
            self.terminal(low_sequence[:, i:i + self.chunk_size])
            for i in range(0, length, self.chunk_size)
        ], dim=1)

    def scan_one(self, detail_sequence, low_sequence, scan):
        """One directed set of lines; shape [batch_of_lines, length, channels]."""
        if detail_sequence.shape != low_sequence.shape:
            raise ValueError("Detail and LL sequences must be aligned")
        x = detail_sequence.float()
        ll = low_sequence.float()
        summaries = self.summaries(ll)
        if self.mode == "local_feature":
            # Same spatial LL summaries as state refresh, supplied as token input.
            context = torch.cat([
                summaries[:, j:j + 1].expand(
                    -1, min(self.chunk_size, x.shape[1] - j * self.chunk_size), -1)
                for j in range(summaries.shape[1])
            ], dim=1)
            x = x + CONTEXT_SCALE * context
            initial = x.new_zeros(x.shape[0], scan.inner, scan.state_channels)
            return scan_with_refresh(scan, x, initial)
        target = self.initial_state(
            (CONTEXT_SCALE * summaries).reshape(-1, summaries.shape[-1]), scan)
        target = target.reshape(
            x.shape[0], summaries.shape[1], scan.inner, scan.state_channels)
        return scan_with_refresh(
            scan, x, target[:, 0], target, self.chunk_size, self.carry)

    def scan_lines(self, detail, low, scan):
        outputs = []
        for reverse in (False, True):
            x = detail.flip(2) if reverse else detail
            coarse = low.flip(2) if reverse else low
            batch, lines, length, channels = x.shape
            x = x.reshape(batch * lines, length, channels)
            coarse = coarse.reshape(batch, lines, length, channels)
            if self.mode == "local_shuffled":
                # Change LL spatial correspondence, but preserve its histogram.
                coarse = coarse.roll(max(1, lines // 2), dims=1)
            coarse = coarse.reshape(batch * lines, length, channels)
            result = self.scan_one(x, coarse, scan).reshape(
                batch, lines, length, channels)
            outputs.append(result.flip(2) if reverse else result)
        return 0.5 * (outputs[0] + outputs[1])


class LocalRefreshFSCMambaE2(FSCMambaE2):
    """Replace only the FSC LL-to-SSM conditioning; keep E2 base/detail paths."""

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown local FSC mode: {mode}")
        super().__init__("state")
        self.variant = VARIANT_PREFIX + mode
        self.detail_fusion.ssm = LocalRefreshTransfer(self.detail_fusion.ssm, mode)
