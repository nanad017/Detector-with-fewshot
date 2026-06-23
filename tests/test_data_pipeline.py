import importlib.util
import json
import sqlite3
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


class SorelFamilyExtractionTests(unittest.TestCase):
    def install_fake_sorel_family_dependencies(self):
        extractor_module = types.ModuleType("thrember.features")

        class FakeExtractor:
            dim = 1

            def feature_vector(self, bytez):
                if bytez == b"bad":
                    raise ValueError("invalid PE")
                return np.array([bytez[0]], dtype=np.float32)

        extractor_module.PEFeatureExtractor = FakeExtractor
        package = types.ModuleType("thrember")
        lmdb_module = types.ModuleType("lmdb")
        msgpack_module = types.ModuleType("msgpack")

        class FakeTransaction:
            def __init__(self, store):
                self.store = store

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def put(self, key, value):
                self.store[key] = value

        class FakeEnvironment:
            def __init__(self):
                self.store = {}

            def begin(self, write=False):
                return FakeTransaction(self.store)

            def sync(self):
                pass

            def close(self):
                pass

        lmdb_module.open = lambda *args, **kwargs: FakeEnvironment()
        msgpack_module.packb = lambda value, use_bin_type=True: repr(value).encode("utf-8")

        old_modules = {
            "thrember": sys.modules.get("thrember"),
            "thrember.features": sys.modules.get("thrember.features"),
            "lmdb": sys.modules.get("lmdb"),
            "msgpack": sys.modules.get("msgpack"),
        }
        sys.modules["thrember"] = package
        sys.modules["thrember.features"] = extractor_module
        sys.modules["lmdb"] = lmdb_module
        sys.modules["msgpack"] = msgpack_module
        return old_modules

    def restore_modules(self, old_modules):
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    def test_failed_feature_does_not_shift_family_labels_and_includes_benign(self):
        old_modules = self.install_fake_sorel_family_dependencies()
        try:
            sorel_family_data = load_module(
                "sorel_family_data", ROOT / "reproduction/sorel_family_data.py"
            )
            from common import DatasetPaths

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                for split in ("train", "test"):
                    malware_a = root / "Virus" / f"Virus {split}" / "FamilyA"
                    malware_b = root / "Virus" / f"Virus {split}" / "FamilyB"
                    benign = root / "Benign" / f"Benign {split}"
                    malware_a.mkdir(parents=True)
                    malware_b.mkdir(parents=True)
                    benign.mkdir(parents=True)
                    malware_payload = b"a" if split == "train" else b"d"
                    benign_payload = b"c" if split == "train" else b"e"
                    (malware_a / "good.exe").write_bytes(malware_payload)
                    (malware_b / "bad.exe").write_bytes(b"bad")
                    (benign / "good.exe").write_bytes(benign_payload)

                output = root / "output"
                sorel_family_data.prepare_sorel_family_data(DatasetPaths(root), output)

                family_names = json.loads(
                    (output / "family_names.json").read_text(encoding="utf-8")
                )
                self.assertEqual(family_names, ["Benign", "FamilyA", "FamilyB"])

                connection = sqlite3.connect(output / "meta.db")
                rows = connection.execute(
                    "SELECT is_malware, family_label FROM meta ORDER BY rl_fs_t, family_label"
                ).fetchall()
                connection.close()
                self.assertEqual(rows, [(0, 0), (1, 1), (0, 0), (1, 1)])
        finally:
            self.restore_modules(old_modules)

    def test_duplicate_sha_reports_data_leak_before_sqlite_insert(self):
        old_modules = self.install_fake_sorel_family_dependencies()
        try:
            sorel_family_data = load_module(
                "sorel_family_data", ROOT / "reproduction/sorel_family_data.py"
            )
            from common import DatasetPaths

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                train_family = root / "Virus" / "Virus train" / "FamilyA"
                test_family = root / "Virus" / "Virus test" / "FamilyA"
                train_benign = root / "Benign" / "Benign train"
                test_benign = root / "Benign" / "Benign test"
                train_family.mkdir(parents=True)
                test_family.mkdir(parents=True)
                train_benign.mkdir(parents=True)
                test_benign.mkdir(parents=True)
                (train_family / "duplicate.exe").write_bytes(b"same")
                (test_family / "duplicate.exe").write_bytes(b"same")
                (train_benign / "benign.exe").write_bytes(b"a")
                (test_benign / "benign.exe").write_bytes(b"b")

                output = root / "output"
                with self.assertRaisesRegex(ValueError, "Duplicate SHA-256"):
                    sorel_family_data.prepare_sorel_family_data(DatasetPaths(root), output)

                duplicates = json.loads(
                    (output / "duplicate_sha256.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(duplicates), 1)
                self.assertEqual(duplicates[0]["first"]["split"], "train")
                self.assertEqual(duplicates[0]["duplicate"]["split"], "test")
        finally:
            self.restore_modules(old_modules)

    def test_same_split_same_family_duplicate_is_skipped(self):
        old_modules = self.install_fake_sorel_family_dependencies()
        try:
            sorel_family_data = load_module(
                "sorel_family_data", ROOT / "reproduction/sorel_family_data.py"
            )
            from common import DatasetPaths

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                train_family = root / "Virus" / "Virus train" / "FamilyA"
                test_family = root / "Virus" / "Virus test" / "FamilyA"
                train_benign = root / "Benign" / "Benign train"
                test_benign = root / "Benign" / "Benign test"
                train_family.mkdir(parents=True)
                test_family.mkdir(parents=True)
                train_benign.mkdir(parents=True)
                test_benign.mkdir(parents=True)
                (train_family / "sample-a.exe").write_bytes(b"same")
                (train_family / "sample-b.exe").write_bytes(b"same")
                (train_benign / "benign.exe").write_bytes(b"a")
                (test_family / "sample.exe").write_bytes(b"c")
                (test_benign / "benign.exe").write_bytes(b"b")

                output = root / "output"
                sorel_family_data.prepare_sorel_family_data(DatasetPaths(root), output)

                skipped = json.loads(
                    (output / "skipped_duplicate_sha256.json").read_text(encoding="utf-8")
                )
                blocking = json.loads(
                    (output / "duplicate_sha256.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(skipped), 1)
                self.assertEqual(blocking, [])

                connection = sqlite3.connect(output / "meta.db")
                sample_count = connection.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
                connection.close()
                self.assertEqual(sample_count, 4)
        finally:
            self.restore_modules(old_modules)


class SorelFamilyModelTests(unittest.TestCase):
    def test_family_network_outputs_one_logit_per_family(self):
        sys.path.insert(0, str(ROOT / "SOREL-20M"))
        sorel_family_model = load_module(
            "sorel_family_model", ROOT / "reproduction/sorel_family_model.py"
        )
        import torch

        model = sorel_family_model.SorelFamilyNetwork(
            feature_dimension=3,
            num_families=4,
        )
        logits = model(torch.zeros(2, 3))
        self.assertEqual(tuple(logits.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
