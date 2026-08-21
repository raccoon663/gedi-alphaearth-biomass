from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from utilities.guards import assert_safe_predictors, assert_target_not_used
from utilities.radar_benchmark import (PALSAR, REPRESENTATIONS, add_palsar_features,
                                       dn_to_gamma0_db, rfdi_from_dn,
                                       validate_feature_schema, validate_paired_frames)


class PalsarPhysicsTests(unittest.TestCase):
    def test_gamma0_formula(self):
        dn = np.array([1000.0, 10000.0])
        np.testing.assert_allclose(dn_to_gamma0_db(dn), 20*np.log10(dn)-83)

    def test_invalid_dn_masked(self):
        result = dn_to_gamma0_db(np.array([0.0, -1.0, np.nan]))
        self.assertTrue(np.isnan(result).all())

    def test_rfdi_uses_linear_power(self):
        hh, hv = np.array([4.0]), np.array([2.0])
        np.testing.assert_allclose(rfdi_from_dn(hh, hv), [(16-4)/(16+4)])

    def test_feature_transform_is_finite(self):
        result = add_palsar_features(pd.DataFrame({"HH": [1000., 2000.], "HV": [500., 1000.]}))
        self.assertEqual(list(result.columns), PALSAR)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_nonpositive_raw_rejected(self):
        with self.assertRaises(ValueError):
            add_palsar_features(pd.DataFrame({"HH": [0.], "HV": [1.]}))


class IntegrityTests(unittest.TestCase):
    def frame(self, name: str) -> pd.DataFrame:
        data = pd.DataFrame({"shot_number": pd.Series(["1", "2"], dtype="string"),
                             "year": [2020, 2021], "agbd": [10., 20.],
                             "spatial_block_id": ["a", "b"], "spatial_fold": [0, 1]})
        for index, feature in enumerate(REPRESENTATIONS[name]):
            data[feature] = float(index + 1)
        return data

    def test_all_schemas_exclude_shortcuts(self):
        for name, features in REPRESENTATIONS.items():
            self.assertEqual(validate_feature_schema(name, features), features)
            assert_safe_predictors(features)

    def test_paired_frames(self):
        validate_paired_frames({name: self.frame(name) for name in REPRESENTATIONS}, domain="source")

    def test_order_change_rejected(self):
        frames = {name: self.frame(name) for name in REPRESENTATIONS}
        frames["palsar_l"] = frames["palsar_l"].iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            validate_paired_frames(frames, domain="source")

    def test_block_split_rejected(self):
        frames = {name: self.frame(name) for name in REPRESENTATIONS}
        for frame in frames.values():
            frame.loc[1, "spatial_block_id"] = "a"
        with self.assertRaises(RuntimeError):
            validate_paired_frames(frames, domain="source")

    def test_target_information_guard(self):
        with self.assertRaises(RuntimeError):
            assert_target_not_used(stage="model_tuning", target_labels_loaded=True)

    def test_draw_rng_deterministic(self):
        a = np.random.default_rng(42).permutation(100)
        b = np.random.default_rng(42).permutation(100)
        np.testing.assert_array_equal(a, b)

    def test_common_sample_builder_preserves_identical_rows(self):
        import importlib.util
        module_path = ROOT / "scripts/02_data_preparation/build_palsar_common_sample.py"
        spec = importlib.util.spec_from_file_location("common_builder", module_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = pd.DataFrame({"shot_number": pd.Series(["1", "2"], dtype="string"),
                                   "year": [2020, 2021], "agbd": [10., 20.],
                                   "spatial_block_id": ["a", "b"], "spatial_fold": [0, 1]})
            conv = common.copy()
            for feature in set(REPRESENTATIONS["sentinel1_sentinel2"]): conv[feature] = 1.0
            aef = common[["shot_number", "year"]].copy()
            for feature in set(REPRESENTATIONS["alphaearth"]): aef[feature] = 2.0
            palsar = common[["shot_number", "year"]].copy()
            for feature in PALSAR: palsar[feature] = 3.0
            palsar["palsar_rfdi"] = .2; palsar["palsar_valid"] = True
            for name, frame in [("conv", conv), ("aef", aef), ("palsar", palsar)]:
                frame.to_parquet(tmp / f"{name}.parquet", index=False)
            summary = module.build("source", tmp/"conv.parquet", tmp/"aef.parquet",
                                   tmp/"palsar.parquet", tmp/"out", tmp/"freeze.json")
            self.assertEqual(summary["n_after_masking"], 2)
            outputs = {name: pd.read_parquet(tmp/"out"/f"source_{name}.parquet")
                       for name in REPRESENTATIONS}
            validate_paired_frames(outputs, domain="source")


if __name__ == "__main__":
    unittest.main()
