"""Execute the documented example against a real temporary database."""

from pathlib import Path


def test_readme_example(tmp_path, monkeypatch):
    readme = Path(__file__).resolve().parents[2] / "README.md"
    code = readme.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(readme), "exec"), {})
