import json
from datetime import datetime, timezone
from pathlib import Path

from graphify.cli import _stamped_manifest_files
from graphify.detect import save_manifest

out = Path("graphify-out")
detect = json.loads((out / ".graphify_detect.json").read_text(encoding="utf-8"))
extract = json.loads((out / ".graphify_extract.json").read_text(encoding="utf-8"))
root = Path((out / ".graphify_root").read_text(encoding="utf-8-sig").strip())
partial_path = out / ".graphify_partial.txt"
partial = {
    line.strip()
    for line in partial_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
} if partial_path.exists() else set()
corpus = detect.get("all_files") or detect["files"]
manifest_files = _stamped_manifest_files(
    corpus, extract, root, partial_source_files=partial
)
semantic_types = ("document", "paper", "image")
dispatched = {
    file
    for kind, files in detect["files"].items()
    if kind in semantic_types
    for file in files
}
stamped = {file for files in manifest_files.values() for file in files}
cleared = dispatched - stamped
scan = {file for files in corpus.values() for file in files}
save_manifest(
    manifest_files,
    root=root,
    scan_corpus=scan,
    clear_semantic=cleared or None,
)

cost_path = out / "cost.json"
cost = (
    json.loads(cost_path.read_text(encoding="utf-8"))
    if cost_path.exists()
    else {"runs": [], "total_input_tokens": 0, "total_output_tokens": 0}
)
input_tokens = int(extract.get("input_tokens", 0))
output_tokens = int(extract.get("output_tokens", 0))
cost["runs"].append(
    {
        "date": datetime.now(timezone.utc).isoformat(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "files": detect.get("total_files", 0),
        "token_accounting": "unavailable_from_subagent_runtime",
    }
)
cost["total_input_tokens"] += input_tokens
cost["total_output_tokens"] += output_tokens
cost_path.write_text(json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8")
print(
    f"Manifest saved: {sum(len(files) for files in manifest_files.values())} files; "
    f"partial={len(partial)}; token counters={input_tokens}/{output_tokens} (unavailable)"
)
