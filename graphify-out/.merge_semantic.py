import glob
import json
from pathlib import Path

from graphify.cache import (
    drop_unreported_chunk_sources,
    save_semantic_cache,
)

out = Path("graphify-out")
root = Path((out / ".graphify_root").read_text(encoding="utf-8-sig").strip())
prompt = Path(".codex/skills/graphify/references/extraction-spec.md").resolve()
manifest = json.loads((out / ".graphify_chunks.json").read_text(encoding="utf-8"))
chunk_paths = [out / f".graphify_chunk_{number:02d}.json" for number in range(1, len(manifest) + 1)]
missing = [str(path) for path in chunk_paths if not path.is_file()]
if missing:
    raise SystemExit("Missing chunk files:\n" + "\n".join(missing))

covered = {}
all_nodes, all_edges, all_hyperedges = [], [], []
total_in = total_out = 0
for number, (chunk_path, expected_files) in enumerate(zip(chunk_paths, manifest), start=1):
    data = json.loads(chunk_path.read_text(encoding="utf-8"))
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        raise SystemExit(f"Chunk {number:02d} has invalid nodes/edges schema")
    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        raise SystemExit(f"Chunk {number:02d} has invalid coverage schema")
    absent = [path for path in expected_files if path not in coverage]
    if absent:
        raise SystemExit(f"Chunk {number:02d} missing coverage: {absent}")
    for source, count in coverage.items():
        covered[source] = max(covered.get(source, 0), int(count))
    all_nodes.extend(data.get("nodes", []))
    all_edges.extend(data.get("edges", []))
    all_hyperedges.extend(data.get("hyperedges", []))
    total_in += int(data.get("input_tokens", 0))
    total_out += int(data.get("output_tokens", 0))

partial = []
for source, read_count in sorted(covered.items()):
    path = Path(source)
    if not path.is_absolute():
        path = root / source
    if not path.is_file():
        continue
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        actual = len(PdfReader(str(path)).pages)
    elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}:
        actual = 1
    else:
        actual = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
    if actual and read_count < actual:
        partial.append((source, read_count, actual))

unread, dropped = drop_unreported_chunk_sources(out, root)
partial_sources = list(dict.fromkeys([source for source, _, _ in partial] + unread))
(out / ".graphify_partial.txt").write_text("\n".join(partial_sources), encoding="utf-8")
if unread:
    print(f"WARNING: {len(unread)} unread files; dropped {dropped} guessed items")
if partial:
    print(f"WARNING: {len(partial)} partially read files")
    for source, read_count, actual in partial:
        print(f"  {source}: read {read_count} of {actual}")
if not unread and not partial:
    print("Coverage OK - every semantic file was read completely")

semantic_new = {
    "nodes": all_nodes,
    "edges": all_edges,
    "hyperedges": all_hyperedges,
    "input_tokens": total_in,
    "output_tokens": total_out,
}
(out / ".graphify_semantic_new.json").write_text(
    json.dumps(semantic_new, indent=2, ensure_ascii=False), encoding="utf-8"
)
uncached = [
    line
    for line in (out / ".graphify_uncached.txt").read_text(encoding="utf-8").splitlines()
    if line
]
saved = save_semantic_cache(
    all_nodes,
    all_edges,
    all_hyperedges,
    root=root,
    allowed_source_files=uncached,
    prompt_file=str(prompt),
    partial_source_files=partial_sources,
)
cached_path = out / ".graphify_cached.json"
cached = (
    json.loads(cached_path.read_text(encoding="utf-8"))
    if cached_path.exists()
    else {"nodes": [], "edges": [], "hyperedges": []}
)
seen = set()
deduped_nodes = []
for node in cached.get("nodes", []) + all_nodes:
    if node["id"] not in seen:
        seen.add(node["id"])
        deduped_nodes.append(node)
semantic = {
    "nodes": deduped_nodes,
    "edges": cached.get("edges", []) + all_edges,
    "hyperedges": cached.get("hyperedges", []) + all_hyperedges,
    "input_tokens": total_in,
    "output_tokens": total_out,
}
(out / ".graphify_semantic.json").write_text(
    json.dumps(semantic, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(
    f"Semantic: {len(deduped_nodes)} nodes, {len(semantic['edges'])} edges, "
    f"{len(all_hyperedges)} hyperedges; cached {saved} files"
)
