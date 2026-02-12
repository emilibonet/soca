"""
Queue-based position assignment for mans, daus, and laterals.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .data import QueueSpec, MANS_QUEUE_SPECS, DAUS_QUEUE_SPECS, LATERALS_QUEUE_SPECS
from .optimize import global_peripheral_optimization
from .utils import (
    get_logger,
    filter_available_castellers
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
    """Pipeline using global optimization for all peripheral positions."""
    mans_depth = max((getattr(s, 'max_depth', 0) for s in MANS_QUEUE_SPECS.values()), default=0) or mans
    daus_depth = max((getattr(s, 'max_depth', 0) for s in DAUS_QUEUE_SPECS.values()), default=0) or daus
    laterals_depth = max((getattr(s, 'max_depth', 0) for s in LATERALS_QUEUE_SPECS.values()), default=0) or laterals
    
    available = filter_available_castellers(castellers, all_assignments)
    
    result, stats = global_peripheral_optimization(
        mans_specs=MANS_QUEUE_SPECS if include_mans else {},
        daus_specs=DAUS_QUEUE_SPECS if include_daus else {},
        laterals_specs=LATERALS_QUEUE_SPECS if include_laterals else {},
        mans_depth=mans_depth,
        daus_depth=daus_depth,
        laterals_depth=laterals_depth,
        available_castellers=available,
        all_castellers=castellers,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
        use_weight=True
    )
    
    # Update all_assignments
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type in result:
            all_assignments[queue_type] = result[queue_type]
    
    # Summary per type
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type in result:
            tui_log = _get_tui_logger(f"{queue_type.upper()} Summary")
            if tui_log:
                tui_log.info("%s Summary:", queue_type.upper())
                for queue_id, depth_list in result[queue_type].items():
                    filled = len([d for d in depth_list if d and d[0]])
                    tui_log.info("  %s: %d filled", queue_id, filled)
            else:
                logger.info("%s Summary:", queue_type.upper())
                for queue_id, depth_list in result[queue_type].items():
                    filled = len([d for d in depth_list if d and d[0]])
                    logger.info("  %s: %d filled", queue_id, filled)
    
    return result, stats
