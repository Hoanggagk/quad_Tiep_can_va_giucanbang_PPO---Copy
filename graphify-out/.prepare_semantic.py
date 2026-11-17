import json
from pathlib import Path

from graphify.cache import check_semantic_cache

out = Path("graphify-out")
detect = json.loads((out / ".graphify_detect.json").read_text(encoding="utf-8"))
root = Path((out / ".graphify_root").read_text(encoding="utf-8-sig").strip())
prompt = Path(".codex/skills/graphify/references/extraction-spec.md").resolve()
files = [
    file
    for kind in ("document", "paper", "image")
    for file in detect["files"].get(kind, [])
]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(
    files, root=root, prompt_file=str(prompt)
)
cached_path = out / ".graphify_cached.json"
if cached_nodes or cached_edges or cached_hyperedges:
    cached_path.write_text(
        json.dumps(
            {
                "nodes": cached_nodes,
                "edges": cached_edges,
                "hyperedges": cached_hyperedges,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
else:
    cached_path.unlink(missing_ok=True)
(out / ".graphify_uncached.txt").write_text("\n".join(uncached), encoding="utf-8")
for stale in out.glob(".graphify_chunk_*.json"):
    stale.unlink()

image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
paper_exts = {".pdf"}
images = sorted(path for path in uncached if Path(path).suffix.lower() in image_exts)
papers = sorted(path for path in uncached if Path(path).suffix.lower() in paper_exts)
documents = sorted(path for path in uncached if path not in images and path not in papers)

chunks = [[path] for path in images + papers]
current = []
current_lines = 0
current_dir = None
for path in documents:
    file_path = Path(path)
    try:
        lines = sum(1 for _ in file_path.open(encoding="utf-8", errors="replace"))
    except OSError:
        lines = 0
    parent = str(file_path.parent)
    must_flush = current and (
        len(current) >= 25
        or current_lines + lines > 6000
        or lines > 2000
        or parent != current_dir
    )
    if must_flush:
        chunks.append(current)
        current = []
        current_lines = 0
    if not current:
        current_dir = parent
    current.append(path)
    current_lines += lines
    if lines > 2000:
        chunks.append(current)
        current = []
        current_lines = 0
        current_dir = None
if current:
    chunks.append(current)

(out / ".graphify_chunks.json").write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "semantic_files": len(files),
            "cache_hits": len(files) - len(uncached),
            "uncached": len(uncached),
            "chunks": len(chunks),
            "image_chunks": len(images),
            "paper_chunks": len(papers),
            "document_chunks": len(chunks) - len(images) - len(papers),
        }
    )
)
