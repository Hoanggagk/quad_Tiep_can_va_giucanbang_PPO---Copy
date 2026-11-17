import json
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.export import to_json
from graphify.report import generate

out = Path("graphify-out")
extraction = json.loads((out / ".graphify_extract.json").read_text(encoding="utf-8"))
detection = json.loads((out / ".graphify_detect.json").read_text(encoding="utf-8"))
root = (out / ".graphify_root").read_text(encoding="utf-8-sig").strip()
graph = build_from_json(extraction, root=root, directed=False)
if graph.number_of_nodes() == 0:
    raise SystemExit("ERROR: Graph is empty - extraction produced no nodes")
communities = cluster(graph)
cohesion = score_all(graph, communities)
tokens = {
    "input": extraction.get("input_tokens", 0),
    "output": extraction.get("output_tokens", 0),
}
gods = god_nodes(graph)
surprises = surprising_connections(graph, communities)
labels = {community_id: f"Community {community_id}" for community_id in communities}
questions = suggest_questions(graph, communities, labels)
if not to_json(graph, communities, out / "graph.json", force=False):
    raise SystemExit("ERROR: refused to shrink graphify-out/graph.json")
report = generate(
    graph,
    communities,
    cohesion,
    labels,
    gods,
    surprises,
    detection,
    tokens,
    root,
    suggested_questions=questions,
)
(out / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
analysis = {
    "communities": {str(key): value for key, value in communities.items()},
    "cohesion": {str(key): value for key, value in cohesion.items()},
    "gods": gods,
    "surprises": surprises,
    "questions": questions,
}
(out / ".graphify_analysis.json").write_text(
    json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
)
summaries = {}
for community_id, members in communities.items():
    ranked = sorted(
        members,
        key=lambda node_id: graph.degree[node_id] if node_id in graph else 0,
        reverse=True,
    )
    summaries[str(community_id)] = [
        {
            "id": node_id,
            "label": graph.nodes[node_id].get("label", node_id),
            "source_file": graph.nodes[node_id].get("source_file"),
            "degree": graph.degree[node_id],
        }
        for node_id in ranked[:12]
        if node_id in graph
    ]
(out / ".community_summaries.json").write_text(
    json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(
    f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
    f"{len(communities)} communities"
)
