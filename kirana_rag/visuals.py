"""Find and select original figures from the existing visual evidence groups."""

import re
from collections import defaultdict
from dataclasses import dataclass

from .evidence import Evidence
from .retrieval import tokens

VISUAL_WORD = r"(?:charts?|graphs?|plots?|visuals?|figures?|diagrams?|infographics?|kpis?|scan(?:s|ned)?)"
JOINED_VISUAL_WORD = re.compile(rf"\b(any|some|all|the)[_-]?(?={VISUAL_WORD}\b)", re.I)
FIGURE_NUMBER = re.compile(r"\bfigure\s*(\d+)\b", re.I)
DISPLAY_REQUEST = re.compile(
    rf"^(?:(?:can|could|would) you |please )*(?:show|display|open|list|find|identify|see)\b"
    rf"|^(?:is|are) there\b|^(?:does|do)\b.*\b(?:have|contain|include)\b"
    rf"|^(?:which|what) (?:(?:kinds?|types?) of )?{VISUAL_WORD}\b|^can i see\b"
    rf"|^(?:any )?{VISUAL_WORD}(?:\s+in\b|\s*\??$)",
    re.I,
)
ANALYSIS_REQUEST = re.compile(r"\b(?:and|then)\s+(?:explain|compare|calculate|summari[sz]e|tell)\b", re.I)
GENERIC_TERMS = set(tokens(
    "show display open list find identify please me any there our pdf document annual see available have contain include "
    "some all exist appear found kind type data image images that about showing using "
    "chart charts graph graphs plot plots figure figures visual visuals diagram diagrams "
    "infographic infographics KPI KPIs scan scans scanned"
))
PERIOD_TERMS = {"quarterly": "quarter", "monthly": "month", "yearly": "year"}


@dataclass(frozen=True)
class Visual:
    source: Evidence
    title: str
    kind: str
    number: str | None
    terms: frozenset[str]


def visual_terms(text: str) -> frozenset[str]:
    return frozenset(PERIOD_TERMS.get(term, term) for term in tokens(text))


def visual_kind(title: str, source: Evidence) -> str:
    if source.kind == "scan":
        return "scan"
    if re.search(r"\b(?:kpis?|indicators?|infographics?)\b|at a glance", title, re.I):
        return "infographic"
    if re.search(r"\b(?:network|diagram|map|flowchart)\b", title, re.I):
        return "diagram"
    if re.search(r"\b(?:chart|plot|graph|intensity|trend|by (?:month|quarter|year|country|region|category))\b",
                 title, re.I):
        return "chart"
    return "figure"


def catalog(items: list[Evidence]) -> dict[str, Visual]:
    groups = defaultdict(list)
    for item in items:
        if item.group.startswith("visual-"):
            groups[item.group].append(item)
    visuals = {}
    for group, members in groups.items():
        source = next((item for item in members if item.method == "vision"
                       and item.kind in {"figure", "scan"}), None)
        text = "\n".join(item.searchable for item in members)
        number = FIGURE_NUMBER.search(text)
        if source:
            title = source.label.split(" / ", 1)[0]
        elif number and (title := re.search(r"\bfigure\s*\d+[.:]\s*([^\n|]+)", text, re.I)):
            # If vision was unavailable, a readable figure caption can still link the crop.
            source = members[0]
            title = title[1]
        else:
            continue
        title = re.sub(r"^figure\s*\d+[.:]\s*", "", title, flags=re.I)
        visuals[group] = Visual(source, title, visual_kind(title, source),
                                number[1] if number else None, visual_terms(text))
    return visuals


def mentions_visual(question: str) -> bool:
    return bool(re.search(rf"\b{VISUAL_WORD}\b", JOINED_VISUAL_WORD.sub(r"\1 ", question), re.I))


def requested_visuals(question: str, visuals: dict[str, Visual]) -> list[Visual] | None:
    """None means the question needs the agent. An empty list is a visual lookup miss."""
    question = JOINED_VISUAL_WORD.sub(r"\1 ", question)
    if (not mentions_visual(question) or not DISPLAY_REQUEST.search(question)
            or ANALYSIS_REQUEST.search(question)):
        return None
    figure = FIGURE_NUMBER.search(question)
    if figure:
        return [visual for visual in visuals.values() if visual.number == figure[1]]
    kinds = {kind for pattern, kind in (
        (r"\b(?:charts?|graphs?|plots?)\b", "chart"),
        (r"\bdiagrams?\b", "diagram"),
        (r"\b(?:infographics?|kpis?)\b", "infographic"),
        (r"\bscan(?:s|ned)?\b", "scan"),
    ) if re.search(pattern, question, re.I)}
    candidates = [visual for visual in visuals.values()
                  if (visual.kind in kinds if kinds else visual.kind != "scan")]
    topic = visual_terms(question) - GENERIC_TERMS
    if not topic:
        return candidates
    scores = [(len(topic & visual.terms) + 2 * len(topic & visual_terms(visual.title)), visual)
              for visual in candidates]
    best = max((score for score, _ in scores), default=0)
    return [visual for score, visual in scores if score == best and score > 0]
