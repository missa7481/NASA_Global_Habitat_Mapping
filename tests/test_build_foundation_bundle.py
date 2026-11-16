import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_foundation_bundle import (  # noqa: E402
    coverage_features,
    primary_site_rows,
    source_coverage_features,
)


class FoundationBundleTests(unittest.TestCase):
    def setUp(self):
        self.representations = {
            "BioClim19": ["bio1"],
            "Environment41": ["bio1", "urban"],
            "AlphaEarth64": ["A00"],
            "Environment_AlphaEarth105": ["bio1", "urban", "A00"],
        }

    def test_primary_rows_join_features_and_keep_abundance_fields(self):
        metadata = [
            {
                "sample_id": "site-1",
                "primary": "True",
                "target": "1",
                "year": "2024",
                "longitude": "12.5",
                "latitude": "-3",
                "egg_count": "4",
            }
        ]
        features = [
            {
                "sample_id": "site-1",
                "bio1": "22.1",
                "urban": "0.4",
                "A00": "-0.2",
            }
        ]
        rows = primary_site_rows(
            metadata,
            features,
            self.representations,
            expected_primary_count=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["egg_count"], "4")
        self.assertEqual(rows[0]["A00"], "-0.2")

    def test_coverage_is_aggregated_to_cells_not_exact_points(self):
        rows = [
            {
                "sample_id": "a",
                "longitude": "1.5",
                "latitude": "2.5",
                "target": "1",
                "source": "iNaturalist",
            },
            {
                "sample_id": "b",
                "longitude": "4.5",
                "latitude": "3.5",
                "target": "0",
                "source": "other mosquitoes",
            },
        ]
        geojson = coverage_features(rows)
        self.assertEqual(len(geojson["features"]), 1)
        feature = geojson["features"][0]
        self.assertEqual(feature["properties"]["n_sites"], 2)
        self.assertEqual(feature["properties"]["n_presence"], 1)
        self.assertEqual(feature["geometry"]["coordinates"][0][0], [0, 0])

    def test_multi_source_coverage_keeps_candidate_and_validation_roles_separate(self):
        geojson = source_coverage_features(
            training_rows=[
                {
                    "sample_id": "train-1",
                    "longitude": "1",
                    "latitude": "2",
                    "target": "1",
                },
                {
                    "sample_id": "train-2",
                    "longitude": "2",
                    "latitude": "2",
                    "target": "0",
                },
            ],
            mosquito_alert_rows=[
                {
                    "sample_id": "ma-1",
                    "longitude": "3",
                    "latitude": "2",
                    "candidate_eligible": "True",
                },
                {
                    "sample_id": "ma-2",
                    "longitude": "4",
                    "latitude": "2",
                    "candidate_eligible": "False",
                },
            ],
            globe_rows=[
                {
                    "observation_id": "globe-1",
                    "longitude": "4",
                    "latitude": "2",
                    "coordinates_valid": "True",
                    "species_claim": "True",
                    "expert_verified": "False",
                },
                {
                    "observation_id": "globe-invalid",
                    "longitude": "",
                    "latitude": "",
                    "coordinates_valid": "False",
                    "species_claim": "True",
                    "expert_verified": "False",
                },
            ],
            neon_rows=[
                {
                    "sample_id": "neon-1",
                    "longitude": "4",
                    "latitude": "2",
                    "independent_gt10km": "True",
                    "target": "1",
                },
                {
                    "sample_id": "neon-reused",
                    "longitude": "4",
                    "latitude": "2",
                    "independent_gt10km": "False",
                    "target": "1",
                },
            ],
        )
        self.assertEqual(len(geojson["features"]), 1)
        properties = geojson["features"][0]["properties"]
        self.assertEqual(properties["training_sites"], 2)
        self.assertEqual(properties["training_presence"], 1)
        self.assertEqual(properties["training_background"], 1)
        self.assertEqual(properties["mosquito_alert_candidates"], 1)
        self.assertEqual(properties["globe_habitat_observations"], 1)
        self.assertEqual(properties["globe_species_claims"], 1)
        self.assertEqual(properties["globe_verified_species_claims"], 0)
        self.assertEqual(properties["neon_validation_sites"], 1)
        self.assertEqual(properties["neon_detections"], 1)
        self.assertEqual(geojson["features"][0]["geometry"]["type"], "Polygon")


if __name__ == "__main__":
    unittest.main()
