import unittest

import torch
from torch import nn

from predict.main_higher_spectrum_predict import (
    HigherSpectrumPredictModel,
    _ROCmSafeLayerNorm,
    spectrum_loss,
)


class SpectrumLayerNormTest(unittest.TestCase):
    def test_native_checkpoint_parameters_are_preserved(self):
        native = nn.LayerNorm((2, 8), dtype=torch.float64)
        safe = _ROCmSafeLayerNorm((2, 8), dtype=torch.float64)
        with torch.no_grad():
            native.weight.normal_()
            native.bias.normal_()
        safe.load_state_dict(native.state_dict(), strict=True)
        x = torch.randn(3, 2, 8, dtype=torch.float64)
        torch.testing.assert_close(safe(x), native(x))
        self.assertEqual(set(native.state_dict()), set(safe.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_gpu_forward_and_all_gradients_match_cpu(self):
        # Large row counts exercise the faulty RDNA gamma/beta backward path.
        # Finiteness alone would miss its silently incorrect finite gradients.
        generator = torch.Generator().manual_seed(123)
        for dtype in (torch.float32, torch.float64):
            for rows in (1024, 24384):
                for affine, bias in ((True, True), (True, False), (False, False)):
                    with self.subTest(dtype=dtype, rows=rows, affine=affine, bias=bias):
                        cpu = nn.LayerNorm(128, elementwise_affine=affine, bias=bias, dtype=dtype)
                        gpu = _ROCmSafeLayerNorm(128, elementwise_affine=affine, bias=bias, dtype=dtype).cuda()
                        with torch.no_grad():
                            if cpu.weight is not None:
                                cpu.weight.normal_(generator=generator)
                            if cpu.bias is not None:
                                cpu.bias.normal_(generator=generator)
                        gpu.load_state_dict(cpu.state_dict(), strict=True)
                        x = torch.randn(rows, 128, dtype=dtype, generator=generator).requires_grad_()
                        x_gpu = x.detach().cuda().requires_grad_()
                        dy = torch.randn(rows, 128, dtype=dtype, generator=generator) * .01
                        expected = cpu(x)
                        actual = gpu(x_gpu)
                        expected.backward(dy)
                        actual.backward(dy.cuda())
                        comparisons = [(actual.detach().cpu(), expected.detach()), (x_gpu.grad.cpu(), x.grad)]
                        if affine:
                            comparisons.append((gpu.weight.grad.cpu(), cpu.weight.grad))
                        if bias:
                            comparisons.append((gpu.bias.grad.cpu(), cpu.bias.grad))
                        tolerance = dict(rtol=2e-4, atol=3e-5) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-10)
                        for value, reference in comparisons:
                            self.assertTrue(bool(value.isfinite().all()))
                            torch.testing.assert_close(value, reference, **tolerance)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_transformer_affine_gradients_match_analytic_sums(self):
        model = HigherSpectrumPredictModel(
            d_model=128, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=128, dropout=0, input_cutoff=3.0, output_cutoff=4.5,
        ).cuda().train()
        x = torch.stack((torch.linspace(1.0, 3.0, 40), torch.ones(40)), dim=-1).repeat(2, 1, 1).cuda()
        y = torch.stack((torch.linspace(3.001, 4.5, 512), torch.ones(512)), dim=-1).repeat(2, 1, 1).cuda()
        src_pad = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        tgt_pad = torch.zeros(y.shape[:2], dtype=torch.bool, device=y.device)
        captured = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                def capture(layer, inputs, output, name=name):
                    captured[name] = [layer, inputs[0].detach()]
                    output.register_hook(lambda grad, name=name: captured[name].append(grad.detach()))
                module.register_forward_hook(capture)
        loss, _ = spectrum_loss(model, x, y, src_pad, tgt_pad)
        loss.backward()
        self.assertEqual(len(captured), 7)
        for name, (layer, values, dy) in captured.items():
            with self.subTest(layer=name):
                values, dy = values.cpu().double(), dy.cpu().double()
                normalized = (values - values.mean(-1, keepdim=True)) * torch.rsqrt(
                    values.var(-1, unbiased=False, keepdim=True) + layer.eps
                )
                torch.testing.assert_close(layer.weight.grad.cpu().double(), (dy * normalized).sum((0, 1)), rtol=2e-4, atol=3e-5)
                torch.testing.assert_close(layer.bias.grad.cpu().double(), dy.sum((0, 1)), rtol=2e-4, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
