import json
from pathlib import Path

from graphify.detect import detect
from graphify.paths import out_path
from graphify.watch import _read_build_excludes, _read_build_gitignore

root = Path("graphify-out/.graphify_root").read_text(encoding="utf-8-sig").strip()
result = detect(
    Path(root),
    extra_excludes=_read_build_excludes(out_path()),
    gitignore=_read_build_gitignore(out_path()),
)
Path("graphify-out/.graphify_detect.json").write_text(
    json.dumps(result, ensure_ascii=False), encoding="utf-8"
)
print("Detected", result["total_files"], "files")
