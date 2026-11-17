import json
from pathlib import Path

out = Path("graphify-out")
ast = json.loads((out / ".graphify_ast.json").read_text(encoding="utf-8"))
semantic = json.loads((out / ".graphify_semantic.json").read_text(encoding="utf-8"))
for node in semantic["nodes"]:
    node.setdefault("_origin", "semantic")
for edge in semantic["edges"]:
    edge.setdefault("_origin", "semantic")
seen = {node["id"] for node in ast["nodes"]}
nodes = list(ast["nodes"])
for node in semantic["nodes"]:
    if node["id"] not in seen:
        nodes.append(node)
        seen.add(node["id"])
merged = {
    "nodes": nodes,
    "edges": ast["edges"] + semantic["edges"],
    "hyperedges": semantic.get("hyperedges", []),
    "input_tokens": semantic.get("input_tokens", 0),
    "output_tokens": semantic.get("output_tokens", 0),
}
(out / ".graphify_extract.json").write_text(
    json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(
    f"Merged: {len(nodes)} nodes, {len(merged['edges'])} edges "
    f"({len(ast['nodes'])} AST + {len(semantic['nodes'])} semantic before dedupe)"
)
