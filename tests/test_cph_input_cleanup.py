import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import unicodedata


HELPER = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle_CPH/__init__.py"


class InputCleanupTests(unittest.TestCase):
    def run_helper(self, root):
        folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(root))
        output = io.StringIO()
        with patch.dict(sys.modules, {"folder_paths": folder_paths}), contextlib.redirect_stdout(output):
            self.helper = runpy.run_path(str(HELPER))
        return output.getvalue()

    def test_startup_removes_clipspace_files_recursively_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            targets = [root / "clipspace-image.png", nested / "clipspace-large.bin", root / "clipspace-"]
            for target in targets:
                target.write_bytes(b"x" * (128 * 1024 + 1))
            preserved = [root / "image.png", nested / "my-clipspace-image.png", root / "clipspace.png"]
            for path in preserved:
                path.write_bytes(b"keep")
            named_directory = root / "clipspace-folder"
            named_directory.mkdir()
            (named_directory / "keep.png").write_bytes(b"keep")
            log = self.run_helper(root)
            self.assertTrue(all(not path.exists() for path in targets))
            self.assertTrue(all(path.read_bytes() == b"keep" for path in preserved))
            self.assertTrue((named_directory / "keep.png").exists())
            self.assertIn("Deleted: 3 files", log)

    def test_delete_failure_is_logged_and_other_files_are_still_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "clipspace-blocked.png"
            blocked.write_bytes(b"blocked")
            fork = root / "._image.png"
            fork.write_bytes(b"fork")
            remove = os.remove

            def guarded_remove(path):
                if Path(path) == blocked:
                    raise PermissionError("blocked for test")
                return remove(path)

            with patch("os.remove", side_effect=guarded_remove):
                log = self.run_helper(root)
            self.assertTrue(blocked.exists())
            self.assertFalse(fork.exists())
            self.assertIn("Failed to delete clipspace-blocked.png", log)

    def test_existing_resource_fork_limit_and_nfc_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "._small.png"
            large = root / "._large.png"
            small.write_bytes(b"x" * (128 * 1024))
            large.write_bytes(b"x" * (128 * 1024 + 1))
            decomposed = unicodedata.normalize("NFD", "한글.png")
            (root / decomposed).write_bytes(b"image")
            self.run_helper(root)
            self.assertFalse(small.exists())
            self.assertTrue(large.exists())
            self.assertEqual(self.helper["normalize_to_nfc"](decomposed), "한글.png")
            # macOS filesystems can consider NFC/NFD paths identical, invoking
            # the helper's existing collision suffix behavior.
            images = [path for path in root.iterdir()
                      if unicodedata.normalize("NFC", path.name) in ("한글.png", "한글_nfc.png")]
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].read_bytes(), b"image")


if __name__ == "__main__":
    unittest.main()
