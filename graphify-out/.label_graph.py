import json
import re
from pathlib import Path

from graphify.analyze import suggest_questions
from graphify.build import build_from_json
from graphify.export import to_json
from graphify.report import generate


def words(value: str) -> list[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value or "")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return [word for word in value.strip().split() if word]


def community_label(graph, members, used):
    ranked = sorted(
        (node_id for node_id in members if node_id in graph),
        key=lambda node_id: graph.degree[node_id],
        reverse=True,
    )
    if not ranked:
        base = ["Project", "Component"]
        source_words = []
    else:
        node = graph.nodes[ranked[0]]
        base = words(str(node.get("label") or ranked[0]))
        source = Path(str(node.get("source_file") or ""))
        source_words = words(source.stem) + words(source.parent.name)
    if len(base) < 2:
        base.extend(word for word in source_words if word.casefold() not in {item.casefold() for item in base})
    if len(base) < 2:
        base.append("Module")
    candidate = " ".join(base[:5]).title()
    if candidate in used:
        prefix = next(
            (word for word in source_words if word.casefold() not in {item.casefold() for item in base}),
            "Project",
        )
        candidate = " ".join(([prefix] + base)[:5]).title()
    used.add(candidate)
    return candidate


out = Path("graphify-out")
extraction = json.loads((out / ".graphify_extract.json").read_text(encoding="utf-8"))
detection = json.loads((out / ".graphify_detect.json").read_text(encoding="utf-8"))
analysis = json.loads((out / ".graphify_analysis.json").read_text(encoding="utf-8"))
root = (out / ".graphify_root").read_text(encoding="utf-8-sig").strip()
graph = build_from_json(extraction, root=root, directed=False)
communities = {int(key): value for key, value in analysis["communities"].items()}
cohesion = {int(key): value for key, value in analysis["cohesion"].items()}
used = set()
labels = {
    community_id: community_label(graph, members, used)
    for community_id, members in communities.items()
}
questions = suggest_questions(graph, communities, labels)
tokens = {
    "input": extraction.get("input_tokens", 0),
    "output": extraction.get("output_tokens", 0),
}
report = generate(
    graph,
    communities,
    cohesion,
    labels,
    analysis["gods"],
    analysis["surprises"],
    detection,
    tokens,
    root,
    suggested_questions=questions,
)
(out / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
(out / ".graphify_labels.json").write_text(
    json.dumps({str(key): value for key, value in labels.items()}, ensure_ascii=False),
    encoding="utf-8",
)
if not to_json(graph, communities, out / "graph.json", community_labels=labels):
    raise SystemExit("ERROR: refused to rewrite graph.json with community labels")
violations = [
    label for label in labels.values() if not 2 <= len(label.split()) <= 5
]
print(f"Labels: {len(labels)} communities, {len(violations)} length violations")
print("Suggested questions:")
for question in questions:
    print(f"- {question}")
