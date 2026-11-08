from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    id: str
    page: int
    kind: str
    section: str
    label: str
    text: str
    box: tuple[float, float, float, float]
    group: str = ""
    method: str = "pdf_text"

    @property
    def searchable(self) -> str:
        return " | ".join(dict.fromkeys(part for part in (self.section, self.label, self.text) if part))


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
