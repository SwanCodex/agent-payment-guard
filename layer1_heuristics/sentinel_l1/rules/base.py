from dataclasses import dataclass, field
from typing import List, Optional

# severity: "gray" | "step_up" | "block"

@dataclass
class RuleFlag:
    rule_id: str
    code: str
    severity: str
    evidence: str
    span: Optional[List[int]] = None

    def __post_init__(self):
        self.evidence = self.evidence[:300]


@dataclass
class RuleResult:
    flags: List[RuleFlag] = field(default_factory=list)
    forwarded: List[dict] = field(default_factory=list)   # {"source": "R02", "text": "..."}