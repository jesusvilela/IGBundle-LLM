"""
Tests for Epic 15: train_odyssey_v3 training pipeline.

Validates:
1. Stage configs are well-formed
2. MultimodalStreamingDataset yields samples with correct keys
3. Collator handles text-only and mixed batches
4. Text transforms produce correct tensor shapes
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_odyssey_v3 import (
    STAGE_CONFIGS,
    MultimodalStreamingDataset,
    multimodal_collate,
    StageConfig,
)


class TestStageConfigs:
    def test_all_stages_exist(self):
        assert set(STAGE_CONFIGS.keys()) == {"alignment", "instruction", "domain"}

    def test_stage_types(self):
        for name, cfg in STAGE_CONFIGS.items():
            assert isinstance(cfg, StageConfig), f"{name} is not StageConfig"
            assert cfg.max_steps > 0
            assert cfg.grad_accum > 0
            assert cfg.geo_lambda_max >= 0
            assert cfg.geo_ramp_steps > 0
            assert cfg.base_lr > 0
            assert cfg.fiber_lr > 0

    def test_qlora_only_stage2_3(self):
        assert not STAGE_CONFIGS["alignment"].use_qlora
        assert STAGE_CONFIGS["instruction"].use_qlora
        assert STAGE_CONFIGS["domain"].use_qlora


class TestMultimodalDataset:
    def _make_fake_source(self, n=10, has_image=False):
        items = []
        for i in range(n):
            item = {
                "input_ids": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64, dtype=torch.long),
                "labels": torch.randint(0, 1000, (64,)),
            }
            if has_image:
                item["pixel_values"] = torch.randn(3, 384, 384)
            items.append(item)
        return items

    def test_text_only_iteration(self):
        src = self._make_fake_source(5)
        ds = MultimodalStreamingDataset(
            sources=[{"dataset": src, "transform": lambda x: x, "name": "test"}],
            weights=[1.0],
        )
        sample = next(iter(ds))
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (64,)

    def test_weighted_sampling(self):
        src_a = [{"input_ids": torch.zeros(4), "attention_mask": torch.ones(4),
                   "labels": torch.zeros(4), "source": "a"} for _ in range(100)]
        src_b = [{"input_ids": torch.ones(4), "attention_mask": torch.ones(4),
                   "labels": torch.ones(4), "source": "b"} for _ in range(100)]
        ds = MultimodalStreamingDataset(
            sources=[
                {"dataset": src_a, "transform": lambda x: x, "name": "a"},
                {"dataset": src_b, "transform": lambda x: x, "name": "b"},
            ],
            weights=[0.9, 0.1],
        )
        it = iter(ds)
        counts = {"a": 0, "b": 0}
        for _ in range(200):
            s = next(it)
            counts["a" if s["input_ids"].sum() == 0 else "b"] += 1
        # With 0.9/0.1 weights, source A should dominate
        assert counts["a"] > counts["b"], f"Expected A > B, got {counts}"


class TestCollator:
    def test_text_only_batch(self):
        batch = [
            {"input_ids": torch.randint(0, 100, (32,)),
             "attention_mask": torch.ones(32, dtype=torch.long),
             "labels": torch.randint(0, 100, (32,))}
            for _ in range(4)
        ]
        result = multimodal_collate(batch)
        assert result["input_ids"].shape == (4, 32)
        assert result["labels"].shape == (4, 32)
        assert "pixel_values" not in result

    def test_mixed_batch(self):
        batch = [
            {"input_ids": torch.randint(0, 100, (32,)),
             "attention_mask": torch.ones(32, dtype=torch.long),
             "labels": torch.randint(0, 100, (32,)),
             "pixel_values": torch.randn(3, 384, 384)},
            {"input_ids": torch.randint(0, 100, (32,)),
             "attention_mask": torch.ones(32, dtype=torch.long),
             "labels": torch.randint(0, 100, (32,))},
        ]
        result = multimodal_collate(batch)
        assert result["input_ids"].shape == (2, 32)
        assert "pixel_values" in result
        assert result["pixel_values"].shape == (2, 3, 384, 384)
        assert result["has_image"].tolist() == [True, False]

    def test_all_images_batch(self):
        batch = [
            {"input_ids": torch.randint(0, 100, (16,)),
             "attention_mask": torch.ones(16, dtype=torch.long),
             "labels": torch.randint(0, 100, (16,)),
             "pixel_values": torch.randn(3, 224, 224)}
            for _ in range(3)
        ]
        result = multimodal_collate(batch)
        assert result["pixel_values"].shape == (3, 3, 224, 224)
        assert result["has_image"].all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
