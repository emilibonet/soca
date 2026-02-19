"""
AssignmentState — single source of truth for all castell position assignments.

Invariants enforced:
- A casteller name can appear in at most ONE position across the entire castell.
- The internal `_assigned_names` set is always in sync with actual assignments.
- Tronc positions store: {column: (name1, name2, ...)}
- Queue positions store: {queue_id: [(name,), (name,), ...]}

Usage:
    state = AssignmentState()
    state.assign_tronc('baix', 'Rengla', ('Mario',))
    state.assign_queue('mans', 'Rengla', [('Xavi',), ('Luca',)])
    available = state.filter_available(castellers_df)
    all_dict = state.to_dict()   # for optimize.py / display.py compatibility
"""
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
from .utils import get_logger

logger = get_logger(__name__)


class DuplicateAssignmentError(ValueError):
    """Raised when a casteller is assigned to more than one position."""
    pass


class AssignmentState:
    """Single source of truth for all castell position assignments.

    Thread-unsafe (single-threaded pipeline assumed).
    """

    TRONC_POSITIONS = frozenset({
        'baix', 'segon', 'terç', 'crossa', 'contrafort',
        'agulla', 'dosos', 'acotxador', 'enxaneta',
    })
    QUEUE_TYPES = frozenset({'mans', 'daus', 'laterals'})

    def __init__(self) -> None:
        self._assignments: Dict[str, Dict] = {}
        self._assigned_names: Set[str] = set()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_assigned_names(self) -> Set[str]:
        """Return a *copy* of all currently assigned casteller names."""
        return set(self._assigned_names)

    def is_assigned(self, name: str) -> bool:
        """O(1) check whether *name* is already assigned anywhere."""
        return name in self._assigned_names

    def find_assignment(self, name: str) -> Optional[str]:
        """Return a human-readable location string for *name*, or ``None``."""
        for pos, columns in self._assignments.items():
            for col, val in columns.items():
                if self._is_queue(pos):
                    if isinstance(val, list):
                        for depth_idx, depth_tuple in enumerate(val, 1):
                            if isinstance(depth_tuple, tuple) and name in depth_tuple:
                                return f"{pos}[{col}] depth {depth_idx}"
                elif isinstance(val, tuple) and name in val:
                    return f"{pos}[{col}]"
        return None

    def get_position(self, position: str) -> Dict:
        """All column/queue assignments for *position* (returns mutable ref)."""
        return self._assignments.get(position, {})

    def get_tronc_assignment(self, position: str, column: str) -> Tuple:
        """Tuple of names for a tronc position+column (empty tuple if unset)."""
        return self._assignments.get(position, {}).get(column, ())

    def get_queue_assignment(self, queue_type: str, queue_id: str) -> List:
        """Depth-list for a queue (empty list if unset)."""
        return self._assignments.get(queue_type, {}).get(queue_id, [])

    def count_assigned(self, position: str, column: str) -> int:
        """How many non-None names are stored at position[column]."""
        val = self._assignments.get(position, {}).get(column)
        if val is None:
            return 0
        if self._is_queue(position):
            return sum(
                1 for depth_tuple in val
                if isinstance(depth_tuple, tuple) and depth_tuple and depth_tuple[0]
            )
        if isinstance(val, tuple):
            return len([n for n in val if n])
        return 0

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def assign_tronc(
        self,
        position: str,
        column: str,
        names: Tuple[str, ...],
    ) -> None:
        """Assign *names* to a tronc position+column.

        Raises ``DuplicateAssignmentError`` if any name is already assigned
        elsewhere.
        """
        self._check_duplicates(names, position, column)
        self._assignments.setdefault(position, {})
        # If overwriting an existing assignment, untrack old names first
        self._untrack(position, column)
        self._assignments[position][column] = names
        for name in names:
            if name:
                self._assigned_names.add(name)

    def assign_queue(
        self,
        queue_type: str,
        queue_id: str,
        depth_list: List[Tuple[str, ...]],
    ) -> None:
        """Assign a full depth list to a queue position.

        Raises ``DuplicateAssignmentError`` if any name is already assigned
        elsewhere.
        """
        all_names = [
            name
            for depth_tuple in depth_list
            if isinstance(depth_tuple, tuple)
            for name in depth_tuple
            if name
        ]
        self._check_duplicates(all_names, queue_type, queue_id)
        self._assignments.setdefault(queue_type, {})
        # Untrack previous assignment if overwriting
        self._untrack(queue_type, queue_id)
        self._assignments[queue_type][queue_id] = depth_list
        for name in all_names:
            self._assigned_names.add(name)

    def ensure_position(self, position: str) -> None:
        """Ensure an empty dict exists for *position*."""
        self._assignments.setdefault(position, {})

    def clear_position(self, position: str, column: str) -> None:
        """Remove all assignments at position[column] and update tracking."""
        self._untrack(position, column)
        if position in self._assignments:
            self._assignments[position].pop(column, None)

    def remove_name_everywhere(self, name: str) -> None:
        """Remove *name* from every position it appears in."""
        for pos in list(self._assignments.keys()):
            for col in list(self._assignments[pos].keys()):
                val = self._assignments[pos][col]
                if self._is_queue(pos):
                    if isinstance(val, list):
                        new_list = []
                        for dt in val:
                            if isinstance(dt, tuple):
                                cleaned = tuple(n if n != name else None for n in dt)
                                new_list.append(cleaned)
                            else:
                                new_list.append(dt)
                        self._assignments[pos][col] = new_list
                elif isinstance(val, tuple):
                    self._assignments[pos][col] = tuple(
                        n if n != name else None for n in val
                    )
        self._assigned_names.discard(name)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_available(
        self,
        castellers: pd.DataFrame,
        name_col: str = 'Nom complet',
        base_db_unavailable_flag: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return castellers not yet assigned (and optionally not unavailable)."""
        df = castellers[~castellers[name_col].isin(self._assigned_names)]
        if base_db_unavailable_flag and base_db_unavailable_flag in df.columns:
            df = df[~df[base_db_unavailable_flag].astype(bool)]
        return df

    def get_all_assigned_list(self) -> List[str]:
        """Return list of assigned names (for optimize.py compatibility)."""
        return list(self._assigned_names)

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Dict]:
        """Export as a plain dict — used by optimize.py, display.py, JSON output."""
        return self._assignments

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> 'AssignmentState':
        """Import from a plain dict (reconstruct _assigned_names)."""
        state = cls()
        state._assignments = data
        state._rebuild_names()
        return state

    def merge_queue_results(self, result: Dict[str, Dict]) -> None:
        """Merge optimization results for queues into state.

        *result* has structure ``{queue_type: {queue_id: [(name,), ...]}}``.
        Only merges queues not already fully assigned.
        """
        for queue_type in ('mans', 'daus', 'laterals'):
            if queue_type not in result:
                continue
            self._assignments.setdefault(queue_type, {})
            for queue_id, depth_list in result[queue_type].items():
                # Untrack old (if any) and write new
                self._untrack(queue_type, queue_id)
                self._assignments[queue_type][queue_id] = depth_list
                for dt in depth_list:
                    if isinstance(dt, tuple):
                        for name in dt:
                            if name:
                                self._assigned_names.add(name)

    def clean_peripheral_for_tronc(self) -> None:
        """Remove names assigned to tronc positions from peripheral queues.

        This prevents someone preassigned to both baix *and* mans from
        appearing in both.
        """
        tronc_names: Set[str] = set()
        for pos in self.TRONC_POSITIONS:
            if pos not in self._assignments:
                continue
            for col, val in self._assignments[pos].items():
                if isinstance(val, tuple):
                    tronc_names.update(n for n in val if n)

        for queue_type in self.QUEUE_TYPES:
            if queue_type not in self._assignments:
                continue
            for queue_id in list(self._assignments[queue_type].keys()):
                depth_list = self._assignments[queue_type][queue_id]
                if not isinstance(depth_list, list):
                    continue
                cleaned = []
                for dt in depth_list:
                    if isinstance(dt, tuple):
                        new_dt = tuple(
                            n if (n and n not in tronc_names) else None
                            for n in dt
                        )
                        # Preserve tuple size but null-out tronc conflicts
                        cleaned.append(new_dt)
                    else:
                        cleaned.append(dt)
                # If all depths are now empty, remove the queue key
                if all(
                    (not d or d == () or all(x is None for x in d))
                    for d in cleaned
                ):
                    self._untrack(queue_type, queue_id)
                    del self._assignments[queue_type][queue_id]
                else:
                    self._assignments[queue_type][queue_id] = cleaned

        # Rebuild names to stay in sync after cleanup
        self._rebuild_names()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_queue(self, position: str) -> bool:
        return position in self.QUEUE_TYPES

    def _check_duplicates(
        self, names, position: str, column: str
    ) -> None:
        """Raise DuplicateAssignmentError if any name is already assigned."""
        for name in names:
            if not name:
                continue
            if name in self._assigned_names:
                existing = self.find_assignment(name)
                # Allow overwriting the SAME slot
                if existing and existing.startswith(f"{position}[{column}]"):
                    continue
                raise DuplicateAssignmentError(
                    f"'{name}' already assigned to {existing}, "
                    f"cannot assign to {position}[{column}]"
                )

    def _untrack(self, position: str, column: str) -> None:
        """Remove names at position[column] from _assigned_names."""
        val = self._assignments.get(position, {}).get(column)
        if val is None:
            return
        if self._is_queue(position) and isinstance(val, list):
            for dt in val:
                if isinstance(dt, tuple):
                    for name in dt:
                        if name:
                            self._assigned_names.discard(name)
        elif isinstance(val, tuple):
            for name in val:
                if name:
                    self._assigned_names.discard(name)

    def _rebuild_names(self) -> None:
        """Recompute _assigned_names from scratch."""
        self._assigned_names.clear()
        for pos, columns in self._assignments.items():
            for col, val in columns.items():
                if self._is_queue(pos) and isinstance(val, list):
                    for dt in val:
                        if isinstance(dt, tuple):
                            for name in dt:
                                if name:
                                    self._assigned_names.add(name)
                elif isinstance(val, tuple):
                    for name in val:
                        if name:
                            self._assigned_names.add(name)

    def __repr__(self) -> str:
        tronc_count = sum(
            self.count_assigned(pos, col)
            for pos in self._assignments
            if not self._is_queue(pos)
            for col in self._assignments[pos]
        )
        queue_count = sum(
            self.count_assigned(pos, col)
            for pos in self._assignments
            if self._is_queue(pos)
            for col in self._assignments[pos]
        )
        return (
            f"<AssignmentState tronc={tronc_count} queue={queue_count} "
            f"total={len(self._assigned_names)} names>"
        )
