from dataclasses import dataclass
from typing import List, Dict, Union, Literal

# --- Base Event Types ---
@dataclass
class MurderEvent:
    type: Literal['murder']
    name: str

@dataclass
class ShieldEvent:
    type: Literal['shield']
    names: List[str]

@dataclass
class DaggerEvent:
    type: Literal['dagger']
    names: List[str]

@dataclass
class RoundTableEvent:
    type: Literal['round table']
    votes: Dict[str, str]

@dataclass
class BreakfastOrdering:
    group: List[str]

@dataclass
class BreakfastEvent:
    type: Literal['breakfast']
    ordering: List[BreakfastOrdering]

GameEvent = Union[MurderEvent, ShieldEvent, RoundTableEvent, BreakfastEvent, DaggerEvent]

# --- Round ---
@dataclass
class GameRound:
    events: List[GameEvent]

# --- Game Root ---
@dataclass
class GameData:
    participants: List[str]
    traitors: List[str]
    rounds: List[GameRound]
