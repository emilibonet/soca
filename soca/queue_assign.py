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

def assign_queue_positions_global(
    queue_type: str,
    queue_specs: Dict[str, QueueSpec],
    max_depth: int,
    castellers: pd.DataFrame,
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    all_assignments: Dict[str, Dict],
    use_weight: bool = True
) -> Tuple[Dict[str, List[Tuple[str, ...]]], Dict]:
    """Assign all queue positions using global optimization."""
    
    logger.info("Assigning %s QUEUES (global optimization)", queue_type.upper())
    
    # Initialize queue structure
    if queue_type not in all_assignments:
        all_assignments[queue_type] = {}
    
    for queue_id in queue_specs.keys():
        if queue_id not in all_assignments[queue_type]:
            all_assignments[queue_type][queue_id] = []
    
    # Get available candidates once
    available = filter_available_castellers(castellers, all_assignments)
    
    if len(available) == 0:
        logger.warning("No available castellers for %s queues", queue_type)
        return all_assignments[queue_type]
    
    # Run global optimization
    optimal_assignment, opt_stats = adaptive_simulated_annealing_queue_assignment(
        queue_specs=queue_specs,
        max_depth=max_depth,
        available_castellers=available,
        all_castellers=castellers,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
        use_weight=use_weight
    )

    # Store results
    all_assignments[queue_type] = optimal_assignment
    
    # Summary
    logger.info("%s Summary:", queue_type.upper())
    for queue_id in queue_specs.keys():
        filled = len([d for d in all_assignments[queue_type][queue_id] if d and d[0]])
        total = len(all_assignments[queue_type][queue_id])
        logger.info("  %s: %d filled / %d total", queue_id, filled, total)
    
    return all_assignments[queue_type], {'final_score': opt_stats.get('final_score') if isinstance(opt_stats, dict) else opt_stats, 'iterations': opt_stats.get('iterations') if isinstance(opt_stats, dict) else None}

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
    # Get max depths from specs or config
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
            logger.info("%s Summary:", queue_type.upper())
            for queue_id, depth_list in result[queue_type].items():
                filled = len([d for d in depth_list if d and d[0]])
                logger.info("  %s: %d filled", queue_id, filled)
    
    return result, stats
