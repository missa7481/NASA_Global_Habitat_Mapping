import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_fusion_benchmark import (  # noqa: E402
    spatial_groups,
    spatial_oof_probabilities,
)


class FusionBenchmarkTests(unittest.TestCase):
    def test_spatial_groups_use_five_degree_cells(self):
        rows = [
            {"sample_id": "a", "longitude": "0", "latitude": "0"},
            {"sample_id": "b", "longitude": "4.9", "latitude": "4.9"},
            {"sample_id": "c", "longitude": "5", "latitude": "0"},
        ]
        self.assertEqual(spatial_groups(rows).tolist(), ["36:18", "36:18", "37:18"])

    def test_late_fusion_experts_return_complete_spatial_oof_scores(self):
        rows = []
        features = []
        labels = []
        for cell in range(8):
            for sample in range(4):
                rows.append(
                    {
                        "sample_id": f"{cell}-{sample}",
                        "longitude": str(cell * 5),
                        "latitude": "0",
                    }
                )
                label = sample % 2
                labels.append(label)
                features.append([float(label), float(cell) / 8])

        scores = spatial_oof_probabilities(
            np.asarray(features, dtype=float),
            np.asarray(labels, dtype=np.int64),
            spatial_groups(rows),
            (4,),
            random_state=17,
        )
        self.assertEqual(scores.shape, (32,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
