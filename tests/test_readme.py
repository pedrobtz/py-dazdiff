"""The README's Python examples run as written."""

import re
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_python_blocks_run(monkeypatch, capsys):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    assert len(blocks) >= 3
    monkeypatch.chdir(ROOT)
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for block in blocks:
            exec(compile(block, "README.md", "exec"), namespace)
    assert "dazdiff report" in capsys.readouterr().out
