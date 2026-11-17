import json
from pathlib import Path

from graphify.extract import collect_files, extract

out = Path("graphify-out")
detect = json.loads((out / ".graphify_detect.json").read_text(encoding="utf-8"))
root = Path((out / ".graphify_root").read_text(encoding="utf-8-sig").strip())
code_files = []
for file in detect.get("files", {}).get("code", []):
    path = Path(file)
    code_files.extend(collect_files(path) if path.is_dir() else [path])
if code_files:
    result = extract(code_files, cache_root=root, root=root)
else:
    result = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}
(out / ".graphify_ast.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f'AST: {len(result["nodes"])} nodes, {len(result["edges"])} edges')
