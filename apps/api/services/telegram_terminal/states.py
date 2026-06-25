"""Estados del terminal interactivo."""

from enum import StrEnum


class TerminalState(StrEnum):
    EXPLORATION = "EXPLORATION"
    MATCH_SELECTED = "MATCH_SELECTED"
    OPPORTUNITIES_VIEW = "OPPORTUNITIES_VIEW"
    PARLAY_VIEW = "PARLAY_VIEW"
    ANALYSIS_VIEW = "ANALYSIS_VIEW"
