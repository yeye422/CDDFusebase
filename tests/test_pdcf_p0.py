import torch
import torch.nn as nn

from pdcf_model import PDCFModel
from pdcf_modules import BranchNormalizer, FeatureRefactor, PDCFCore


class ToyEncoder(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.base = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.detail = nn.Conv2d(1, channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        base = self.base(inputs)
        detail = self.detail(inputs)
        return base, detail, torch.cat((base, detail), dim=1)


class ToyDecoder(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.output = nn.Conv2d(2 * channels, 1, kernel_size=1)

    def forward(self, inputs, base, detail):
        feature = torch.cat((base, detail), dim=1)
        return torch.sigmoid(self.output(feature) + inputs), feature


def build_toy_model():
    channels = 4
    return PDCFModel(
        encoder=ToyEncoder(channels),
        decoder=ToyDecoder(channels),
        base_fuse=nn.Conv2d(channels, channels, kernel_size=1),
        detail_fuse=nn.Conv2d(channels, channels, kernel_size=1),
        core=PDCFCore(
            channels=channels,
            predictor_hidden_channels=8,
            work_channels=8,
        ),
    )


def test_refactor_identity_initialization_and_inverse():
    torch.manual_seed(1)
    module = FeatureRefactor(channels=4, hidden_channels=8)
    base = torch.randn(2, 4, 9, 7)
    detail = torch.randn(2, 4, 9, 7)

    new_base, new_detail = module(base, detail)
    assert torch.equal(new_base, base)
    assert torch.equal(new_detail, detail)

    with torch.no_grad():
        module.detail_to_base[-1].weight.normal_(0.0, 0.02)
        module.base_to_detail[-1].weight.normal_(0.0, 0.02)
    new_base, new_detail = module(base, detail)
    recovered_base, recovered_detail = module.inverse(new_base, new_detail)

    torch.testing.assert_close(recovered_base, base, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(recovered_detail, detail, rtol=1e-5, atol=1e-6)


def test_branch_normalizer_requires_explicit_compatible_statistics():
    normalizer = BranchNormalizer(channels=4)
    mean = torch.zeros(1, 4, 1, 1)
    scale = torch.full((1, 4, 1, 1), 2.0)
    normalizer.set_statistics(mean, scale, mean + 1.0, scale)
    base = torch.full((1, 4, 3, 3), 2.0)
    detail = torch.full((1, 4, 3, 3), 3.0)

    base_n, detail_n = normalizer(base, detail)
    assert torch.equal(base_n, torch.ones_like(base_n))
    assert torch.equal(detail_n, torch.ones_like(detail_n))
    assert normalizer.statistics_ready.item()


def test_pdcf_zero_initialization_is_exact_c0_identity():
    torch.manual_seed(2)
    model = build_toy_model().eval()
    optical = torch.rand(2, 1, 12, 10)
    sar = torch.rand(2, 1, 12, 10)

    with torch.no_grad():
        c0_output, c0_feature = model.forward_c0(optical, sar)
        pdcf_output, pdcf_feature, diagnostics = model(
            optical, sar, return_diagnostics=True
        )

    assert torch.equal(pdcf_output, c0_output)
    assert torch.equal(pdcf_feature, c0_feature)
    assert torch.count_nonzero(diagnostics["delta_base"]) == 0
    assert torch.count_nonzero(diagnostics["delta_detail"]) == 0
    assert torch.equal(diagnostics["fused_base"], diagnostics["c0_base"])
    assert torch.equal(diagnostics["fused_detail"], diagnostics["c0_detail"])


def test_pdcf_freezes_c0_and_keeps_new_core_trainable():
    model = build_toy_model()
    for module in (
        model.encoder,
        model.decoder,
        model.base_fuse,
        model.detail_fuse,
    ):
        assert all(not parameter.requires_grad for parameter in module.parameters())
    assert all(parameter.requires_grad for parameter in model.core.parameters())


def test_initial_gradient_reaches_joint_output_but_not_frozen_c0():
    torch.manual_seed(3)
    model = build_toy_model().train()
    optical = torch.rand(2, 1, 8, 8)
    sar = torch.rand(2, 1, 8, 8)
    output, _ = model(optical, sar)
    output.mean().backward()

    output_layer = model.core.joint_projection.body[-1]
    assert output_layer.weight.grad is not None
    assert torch.count_nonzero(output_layer.weight.grad) > 0
    for module in (
        model.encoder,
        model.decoder,
        model.base_fuse,
        model.detail_fuse,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_spd_condition_contract_is_explicit():
    core = PDCFCore(
        channels=4,
        predictor_hidden_channels=8,
        work_channels=8,
        spd_condition_channels=3,
    )
    tensors = [torch.randn(1, 4, 6, 5) for _ in range(4)]
    condition = torch.randn(1, 3, 3, 3)
    delta_base, delta_detail = core(*tensors, spd_condition=condition)
    assert delta_base.shape == tensors[0].shape
    assert delta_detail.shape == tensors[0].shape

