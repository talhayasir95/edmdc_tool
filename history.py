"""
History Stack of Configs
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy

@dataclass
class History:
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)

    def push(self, config: Dict[str, Any]) -> None:
        self.undo_stack.append(copy.deepcopy(config))
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self) -> Optional[Dict[str, Any]]:
        current = self.undo_stack.pop()
        self.redo_stack.append(current)
        return copy.deepcopy(self.undo_stack[-1])

    def redo(self) -> Optional[Dict[str, Any]]:
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return copy.deepcopy(state)