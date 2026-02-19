"""
Queue-based position assignment for mans, daus, and laterals.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .data import QueueSpec, MANS_QUEUE_SPECS, DAUS_QUEUE_SPECS, LATERALS_QUEUE_SPECS
from .optimize import global_peripheral_optimization
from .utils import (
    get_logger,
    filter_available_castellers,
    _extract_names_from_assignments,
)

logger = get_logger(__name__)

_tui_manager = None

def _get_tui_logger(section_name=None):
    """Get TUI-aware logger if manager is available."""
    if _tui_manager is not None:
        try:
            from .display import SectionLogger
            return SectionLogger(_tui_manager, section_name)
        except ImportError:
            pass
    return None


def _clean_peripheral_preassignments(all_assignments: Dict[str, Dict]) -> None:
    """Remove names already assigned to tronc positions from peripheral queues.

    Operates in-place on *all_assignments*.
    """
    TRONC_POSITIONS = {
        'baix', 'segon', 'terç', 'crossa', 'contrafort',
        'agulla', 'dosos', 'acotxador', 'enxaneta',
    }
    tronc_assigned: set = set()
    for pos_name in TRONC_POSITIONS:
        if pos_name not in all_assignments:
            continue
        for _col, assignment in all_assignments[pos_name].items():
            if isinstance(assignment, tuple):
                tronc_assigned.update(name for name in assignment if name)

    for queue_type in ('mans', 'daus', 'laterals'):
        if queue_type not in all_assignments:
            continue
        for queue_id in list(all_assignments[queue_type].keys()):
            depth_list = all_assignments[queue_type][queue_id]
            if not isinstance(depth_list, list):
                continue
            cleaned_depths = []
            for depth_tuple in depth_list:
                if isinstance(depth_tuple, tuple):
                    cleaned = tuple(
                        name if (name and name not in tronc_assigned) else None
                        for name in depth_tuple
                    )
                    cleaned_depths.append(cleaned)
                else:
                    cleaned_depths.append(depth_tuple)
            # Drop key entirely when all depths are now empty
            if all(
                (not d or d == () or all(x is None for x in d))
                for d in cleaned_depths
            ):
                del all_assignments[queue_type][queue_id]
            else:
                all_assignments[queue_type][queue_id] = cleaned_depths


def assign_rows_pipeline(
    castellers: pd.DataFrame,
    columns: Dict[str, float],
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    all_assignments: Dict[str, Dict],
    mans: int = 3,
    daus: int = 3,
    laterals: int = 5,
    include_laterals: bool = True,
    include_daus: bool = True,
    include_mans: bool = True
):
    """Pipeline using global optimization for all peripheral positions, preserving preassignments.

    Key behaviours:
    - Computes available castellers by excluding only TRONC-assigned people
      (peripheral-preassigned people remain available for optimization).
    - Extracts locked positions from peripheral preassignments so the
      optimizer preserves them.
    - Merges results at depth level (preassigned depths take precedence).
    """
    mans_depth = max((getattr(s, 'max_depth', 0) for s in MANS_QUEUE_SPECS.values()), default=0) or mans
    daus_depth = max((getattr(s, 'max_depth', 0) for s in DAUS_QUEUE_SPECS.values()), default=0) or daus
    laterals_depth = max((getattr(s, 'max_depth', 0) for s in LATERALS_QUEUE_SPECS.values()), default=0) or laterals

    # Clean peripheral preassignments (remove tronc conflicts)
    _clean_peripheral_preassignments(all_assignments)

    # Helper to check if a queue is already fully assigned
    def is_queue_assigned(queue_type: str, queue_id: str, required_depth: int) -> bool:
        if queue_type not in all_assignments:
            return False
        if queue_id not in all_assignments[queue_type]:
            return False
        assignments = all_assignments[queue_type][queue_id]
        if not isinstance(assignments, list):
            return False
        if len(assignments) < required_depth:
            return False
        for i in range(required_depth):
            depth_tuple = assignments[i]
            if not depth_tuple or depth_tuple == () or depth_tuple == (None,):
                return False
        return True
    
    mans_specs_to_optimize = {
        qid: spec for qid, spec in MANS_QUEUE_SPECS.items()
        if not is_queue_assigned('mans', qid, mans_depth)
    } if include_mans else {}

    daus_specs_to_optimize = {
        qid: spec for qid, spec in DAUS_QUEUE_SPECS.items()
        if not is_queue_assigned('daus', qid, daus_depth)
    } if include_daus else {}

    laterals_specs_to_optimize = {
        qid: spec for qid, spec in LATERALS_QUEUE_SPECS.items()
        if not is_queue_assigned('laterals', qid, laterals_depth)
    } if include_laterals else {}
    
    # Skip optimization if nothing to optimize
    if not any([mans_specs_to_optimize, daus_specs_to_optimize, laterals_specs_to_optimize]):
        logger.info("All peripheral positions already assigned via preassignments")
        return {}, {'total_score': 0.0}
    
    # Compute available castellers excluding TRONC assignments AND all
    # peripheral-preassigned names.  We must exclude peripheral names
    # here because fully-preassigned queues are skipped by the optimiser
    # (via is_queue_assigned) and their names would otherwise leak into
    # the candidate pool, causing duplicates.
    PERIPHERAL_TYPES = {'mans', 'daus', 'laterals'}
    tronc_only = {k: v for k, v in all_assignments.items() if k not in PERIPHERAL_TYPES}
    available = filter_available_castellers(castellers, tronc_only)

    # Collect every name already preassigned to a peripheral queue
    peripheral_preassigned_names: set = set()
    for queue_type in PERIPHERAL_TYPES:
        if queue_type not in all_assignments:
            continue
        for _qid, depth_list in all_assignments[queue_type].items():
            if not isinstance(depth_list, list):
                continue
            for dt in depth_list:
                if isinstance(dt, tuple):
                    for name in dt:
                        if name:
                            peripheral_preassigned_names.add(name)

    if peripheral_preassigned_names:
        available = available[~available['Nom complet'].isin(peripheral_preassigned_names)]

    # Build locked_positions: {spec_key: {depth_idx: (name,)}}
    # These are depths that already have preassigned castellers and must
    # not be overwritten by the optimizer.
    all_specs_keys: Dict[str, str] = {}   # spec_key -> queue_type
    for qid in mans_specs_to_optimize:
        all_specs_keys[f'mans:{qid}'] = 'mans'
    for qid in daus_specs_to_optimize:
        all_specs_keys[f'daus:{qid}'] = 'daus'
    for qid in laterals_specs_to_optimize:
        all_specs_keys[f'laterals:{qid}'] = 'laterals'

    locked_positions: Dict[str, Dict[int, Tuple[str, ...]]] = {}
    for spec_key, queue_type in all_specs_keys.items():
        _, queue_id = spec_key.split(':', 1)
        if queue_type in all_assignments and queue_id in all_assignments[queue_type]:
            prev = all_assignments[queue_type][queue_id]
            if isinstance(prev, list):
                key_locked: Dict[int, Tuple[str, ...]] = {}
                for i, dt in enumerate(prev):
                    if isinstance(dt, tuple) and dt and dt[0]:
                        key_locked[i] = dt
                if key_locked:
                    locked_positions[spec_key] = key_locked

    result, stats = global_peripheral_optimization(
        mans_specs=mans_specs_to_optimize,
        daus_specs=daus_specs_to_optimize,
        laterals_specs=laterals_specs_to_optimize,
        mans_depth=mans_depth,
        daus_depth=daus_depth,
        laterals_depth=laterals_depth,
        available_castellers=available,
        all_castellers=castellers,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
        use_weight=True,
        locked_positions=locked_positions,
    )

    # ----- Depth-level merge: preassigned depths take priority ----------
    merged_result: Dict[str, Dict] = {'mans': {}, 'daus': {}, 'laterals': {}}

    for queue_type in ('mans', 'daus', 'laterals'):
        # Start with optimizer results
        if queue_type in result:
            merged_result[queue_type] = result[queue_type].copy()

        # Merge preassignment depths on top (preassigned wins per-depth)
        if queue_type in all_assignments:
            for queue_id, preassigned_depths in all_assignments[queue_type].items():
                if not isinstance(preassigned_depths, list):
                    continue
                if queue_id not in merged_result[queue_type]:
                    # Queue not optimized at all — keep preassignment as-is
                    merged_result[queue_type][queue_id] = preassigned_depths
                else:
                    # Depth-level merge
                    opt_depths = merged_result[queue_type][queue_id]
                    max_len = max(len(preassigned_depths), len(opt_depths))
                    merged = []
                    for i in range(max_len):
                        pre = preassigned_depths[i] if i < len(preassigned_depths) else (None,)
                        opt = opt_depths[i] if i < len(opt_depths) else (None,)
                        # Preassigned depth wins if it has a real name
                        if isinstance(pre, tuple) and pre and pre[0]:
                            merged.append(pre)
                        else:
                            merged.append(opt)
                    merged_result[queue_type][queue_id] = merged

    return merged_result, stats