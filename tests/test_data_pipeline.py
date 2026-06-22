import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduction"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataSplitTests(unittest.TestCase):
    def test_original_deepmd_split_rejects_zero_test_size(self):
        dataset = load_module(
            "deepmd_dataset",
            ROOT / "deep-malware-detection/src/deep_malware_detection/dataset.py",
        )
        with self.assertRaises(ValueError):
            dataset.train_val_test_split(range(20), 0.2, 0.0)


class EmberExtractionTests(unittest.TestCase):
    def test_failed_feature_does_not_shift_labels(self):
        extractor_module = types.ModuleType("thrember.features")

        class FakeExtractor:
            def feature_vector(self, bytez):
                if bytez == b"bad":
                    raise ValueError("invalid PE")
                return np.array([bytez[0]], dtype=np.float32)

        extractor_module.PEFeatureExtractor = FakeExtractor
        package = types.ModuleType("thrember")

        old_package = sys.modules.get("thrember")
        old_features = sys.modules.get("thrember.features")
        sys.modules["thrember"] = package
        sys.modules["thrember.features"] = extractor_module
        try:
            ember_data = load_module("ember_data", ROOT / "reproduction/ember_data.py")
            from common import DatasetPaths

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                for split in ("train", "test"):
                    malware = root / "Virus" / f"Virus {split}" / "FamilyA"
                    benign = root / "Benign" / f"Benign {split}"
                    malware.mkdir(parents=True)
                    benign.mkdir(parents=True)
                    (malware / "good.exe").write_bytes(b"a")
                    (benign / "good.exe").write_bytes(b"c")
                (root / "Virus" / "Virus train" / "FamilyA" / "bad.exe").write_bytes(b"bad")

                output = root / "output"
                ember_data.prepare_ember_data(
                    DatasetPaths(root),
                    output,
                    "binary",
                )
                np.testing.assert_array_equal(
                    np.memmap(output / "y_train.dat", dtype=np.int32, mode="r"),
                    np.array([1, 0]),
                )
        finally:
            if old_package is None:
                sys.modules.pop("thrember", None)
            else:
                sys.modules["thrember"] = old_package
            if old_features is None:
                sys.modules.pop("thrember.features", None)
            else:
                sys.modules["thrember.features"] = old_features


if __name__ == "__main__":
    unittest.main()
