"""Check release wheels contain profiles/notices and no media or model weights."""

import sys
import zipfile
from pathlib import Path

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
wheels = list(folder.glob("*.whl"))
assert wheels, "No wheel found"
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in ("TEMPLATE.toml", "example_show.toml"):
            assert f"peiyin/data/profiles/{name}" in names, name
        for name in ("LICENSE", "NOTICE.md", "DISCLAIMER.md", "docs/MODEL-PROVENANCE.md"):
            assert any(n.endswith("/licenses/" + name) for n in names), name
        forbidden = {".mp4", ".mp3", ".wav", ".srt", ".onnx", ".mkv", ".mov"}
        assert not [n for n in names if Path(n).suffix.lower() in forbidden]
    print(f"PASS: {wheel.name} contains profiles/notices and no media or weights.")
