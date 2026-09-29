import unittest

import torch
from torch.utils.data import DataLoader

from predict.main_higher_spectrum_predict import (
    HigherSpectrumPredictModel,
    SpectrumPredictDataset,
    predict_batch,
    spectrum_loss,
    test as evaluate,
    train,
)


class SpectrumMixedPrecisionTest(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.options = dict(
            d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=64, dropout=0, input_cutoff=3.0, output_cutoff=4.5,
        )
        # Include adjacent dimensions that both collapse to 4.5 in BF16/FP16,
        # signed coefficients outside BF16's exact integer range, and padding.
        self.pairs = [
            ([[1.5125, 16005], [3.0, -4400]],
             [[4.499, 16005], [4.4995, -4400], [4.5, 3]]),
            ([[2.0, 2]], []),
        ]
        self.batch = tuple(
            t.to(self.device) for t in SpectrumPredictDataset.collate_spectra(self.pairs)
        )

    def model(self, use_amp=True):
        return HigherSpectrumPredictModel(**self.options, use_amp=use_amp).to(self.device)

    def test_backbone_uses_bf16_and_heads_and_labels_keep_float32(self):
        model = self.model().train()
        dtypes = {}
        modules = {
            "source_embedding": model.src_embed[0],
            "target_embedding": model.tgt_embed[0],
            "encoder_linear": model.encoder.layers[0].linear1,
            "decoder_linear": model.decoder.layers[0].linear1,
            "gap": model.gap_head,
            "count": model.count_head[0],
        }
        handles = [
            module.register_forward_hook(
                lambda module, inputs, output, name=name:
                dtypes.update({name: (inputs[0].dtype, output.dtype)})
            )
            for name, module in modules.items()
        ]
        x, y, src_pad, tgt_pad = self.batch
        original = tuple(t.clone() for t in self.batch)
        out = model(x, y, src_pad, tgt_pad)
        counts = model.multiplicity_parameters(out["hidden"][:, :-1], y[..., 0])
        for handle in handles:
            handle.remove()
        for name in ("source_embedding", "target_embedding", "encoder_linear", "decoder_linear"):
            self.assertEqual(dtypes[name][1], torch.bfloat16)
        for name in ("gap", "count"):
            self.assertEqual(dtypes[name], (torch.float32, torch.float32))
        for value in (*out.values(), *counts.values()):
            if value.is_floating_point():
                self.assertEqual(value.dtype, torch.float32)
        torch.testing.assert_close(out["previous_dimensions"][:, 1:], y[..., 0], rtol=0, atol=0)
        for before, after in zip(original, self.batch):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_loss_and_gradients_match_float32_and_checkpoint_loads(self):
        reference = self.model(use_amp=False).train()
        mixed = self.model().train()
        mixed.load_state_dict(reference.state_dict(), strict=True)
        loss_ref, parts_ref = spectrum_loss(reference, *self.batch)
        loss_amp, parts_amp = spectrum_loss(mixed, *self.batch)
        self.assertEqual(loss_amp.dtype, torch.float32)
        for name in parts_ref:
            self.assertTrue(bool(torch.isfinite(parts_amp[name])))
            torch.testing.assert_close(parts_amp[name], parts_ref[name], rtol=0.03, atol=0.02)
        loss_ref.backward()
        loss_amp.backward()
        ref_grad = torch.cat([p.grad.flatten() for p in reference.parameters() if p.grad is not None])
        amp_grad = torch.cat([p.grad.flatten() for p in mixed.parameters() if p.grad is not None])
        self.assertTrue(bool(torch.isfinite(amp_grad).all()))
        relative_error = (amp_grad - ref_grad).norm() / ref_grad.norm()
        self.assertLess(relative_error.item(), 0.08)

    def test_training_evaluation_and_generation(self):
        model = self.model()
        dataset = SpectrumPredictDataset(
            [x for x, _ in self.pairs], [y for _, y in self.pairs],
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_spectra)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = model.gap_head.weight.detach().clone()
        train(loader, model, optimizer, torch.device(self.device))
        self.assertFalse(torch.equal(before, model.gap_head.weight))
        metrics = evaluate(loader, model, torch.device(self.device))
        self.assertTrue(bool(torch.isfinite(metrics["loss"])))

        # Force a single endpoint prediction, covering BOS-only decoding and
        # both prediction heads without depending on random stopping behavior.
        with torch.no_grad():
            model.stop_head.weight.zero_()
            model.stop_head.bias.fill_(-20)
            model.endpoint_head.weight.zero_()
            model.endpoint_head.bias.fill_(20)
        model.eval()
        x, _, src_pad, _ = self.batch
        prediction, lengths, reasons = predict_batch(model, x, src_pad, max_levels=2)
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertEqual(lengths.tolist(), [1, 1])
        self.assertEqual(reasons, ["upper_cutoff", "upper_cutoff"])
        self.assertTrue(bool((prediction[..., 0] == 4.5).all()))
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_all_empty_targets(self):
        model = self.model()
        empty_batch = tuple(t.to(self.device) for t in SpectrumPredictDataset.collate_spectra([
            ([[1.5125, 1]], []), ([[2.0, -2]], []),
        ]))
        for training in (True, False):
            model.train(training)
            with torch.set_grad_enabled(training):
                loss, parts = spectrum_loss(model, *empty_batch)
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertEqual(loss.dtype, torch.float32)
                for name in ("endpoint", "gap", "sign", "magnitude"):
                    self.assertEqual(parts[name].item(), 0)
                if training:
                    loss.backward()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
    "GPU with native BF16 support required",
)
class SpectrumMixedPrecisionGPUTest(SpectrumMixedPrecisionTest):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
