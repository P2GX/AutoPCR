from typing import TypedDict


class Hit(TypedDict):
    start: int
    end: int
    text: str
    hpo_id: str
    score: float
