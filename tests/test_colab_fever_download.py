"""Exercise notebook download helpers without fetching the 1.7 GB archive."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile


def helpers():
    path = Path(__file__).resolve().parents[1] / "notebooks/05_run_reviewed_pilot_colab.ipynb"
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "def download_fever_archive" in "".join(cell["source"]))
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id.startswith("FEVER_")
                                                    for target in node.targets)]
    namespace = {"zipfile": zipfile}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Response(io.BytesIO):
    def __init__(self, data, size=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if size is None else size)}


class TestColabFeverDownload(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "wiki-pages.zip"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("wiki-pages/wiki-001.jsonl", '{"id":"fixture"}\n')
        self.data = stream.getvalue()
        self.md5 = hashlib.md5(self.data).hexdigest()
        self.download = helpers()["download_fever_archive"]

    def test_404_falls_back_and_promotes_only_a_verified_zip(self):
        error = urllib.error.HTTPError("https://first", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect=[error, Response(self.data)]) as opened:
            self.download(self.target, ("https://first", "https://mirror"), self.md5)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.target.with_suffix(".zip.part").exists())
        metadata = json.loads(self.target.with_suffix(".download.json").read_text())
        self.assertEqual(metadata["source"], "https://mirror")
        self.assertEqual(metadata["sha256"], hashlib.sha256(self.data).hexdigest())

    def test_html_response_or_truncated_download_never_becomes_final_archive(self):
        for response in (Response(b"<html>error</html>"), Response(self.data, len(self.data) + 10)):
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "verified FEVER"):
                    self.download(self.target, ("https://bad",), self.md5)
            self.assertFalse(self.target.exists())

    def test_verified_existing_archive_is_reused_without_network(self):
        self.target.write_bytes(self.data)
        with patch("urllib.request.urlopen", side_effect=AssertionError("No download")):
            self.download(self.target, expected_md5=self.md5)
        self.assertTrue(self.target.with_suffix(".download.json").is_file())

    def test_corrupt_cache_requires_explicit_replacement(self):
        self.target.write_text("not a zip")
        with self.assertRaisesRegex(ValueError, "Cached archive is invalid"):
            self.download(self.target, expected_md5=self.md5)
        self.assertEqual(self.target.read_text(), "not a zip")

    def test_user_runtime_and_output_choices_are_preserved(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/05_run_reviewed_pilot_colab.ipynb"
        notebook = json.loads(path.read_text())
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotIn('urlretrieve("https://s3-eu-west-1.amazonaws.com/fever.ai/', code)
        self.assertIn('"uv", "venv", "--python", "3.12"', code)
        self.assertIn('Path("/content/hallucination_reviewed_pilot")', code)


if __name__ == "__main__":
    unittest.main()
