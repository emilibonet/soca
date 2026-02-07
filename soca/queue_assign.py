"""
Queue-based position assignment for mans, daus, and laterals.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .data import QueueSpec, MANS_QUEUE_SPECS, DAUS_QUEUE_SPECS, LATERALS_QUEUE_SPECS
from .optimize import (
    _get_valid_candidates,
    _calculate_candidate_score,
    _calculate_reference_heights_for_queue,
    create_position_spec_from_queue
)
from .utils import (
    get_logger,
    filter_available_castellers
)

logger = get_logger(__name__)

MAX_QUEUE_DEPTH = 20  # Hard cap to prevent runaway loops

def assign_queue_positions(
    queue_type: str,
    queue_specs: Dict[str, QueueSpec],
    max_depth: int,
    castellers: pd.DataFrame,
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    all_assignments: Dict[str, Dict],
    use_weight: bool = True
) -> Dict[str, List[Tuple[str, ...]]]:
    """Assign queue positions using breadth-first fill."""
    
    logger.info("=== %s QUEUES ===", queue_type.upper())
    
    # Initialize queue structure
    if queue_type not in all_assignments:
        all_assignments[queue_type] = {}
    
    for queue_id in queue_specs.keys():
        if queue_id not in all_assignments[queue_type]:
            all_assignments[queue_type][queue_id] = []
    
    # Breadth-first assignment
    depth = 1
    max_depth = min(max_depth, MAX_QUEUE_DEPTH)
    
    while depth <= max_depth:
        logger.info("Filling depth %d", depth)
        
        assignments_made = 0
        
        for queue_id, queue_spec in queue_specs.items():
            # CRITICAL FIX: Get fresh available list INSIDE the loop for each queue
            available = filter_available_castellers(castellers, all_assignments)
            if len(available) == 0:
                logger.info("No more available castellers at depth %d for %s", depth, queue_id)
                all_assignments[queue_type][queue_id].append((None,))
                continue
            
            # Calculate reference height
            ref_height = _calculate_reference_heights_for_queue(
                queue_spec,
                depth,
                all_assignments,
                column_tronc_heights,
                castellers
            )
            
            # Create position spec
            pos_spec = create_position_spec_from_queue(queue_spec, depth)
            
            # Get valid candidates from FRESH available list
            candidates = _get_valid_candidates(
                available,  # Use fresh available list, not full castellers
                pos_spec,
                allow_relaxed=True,
                requested_count=1
            )
            
            if len(candidates) == 0:
                logger.info("No valid candidates for %s at depth %d", queue_id, depth)
                all_assignments[queue_type][queue_id].append((None,))
                continue
            
            # Score and select
            candidates = candidates.copy()
            candidates['score'] = candidates.apply(
                lambda row: _calculate_candidate_score(
                    row,
                    ref_height,
                    pos_spec,
                    [],
                    use_weight
                ),
                axis=1
            )
            
            best = candidates.nsmallest(1, 'score')
            if not best.empty:
                selected_name = best['Nom complet'].iloc[0]
                all_assignments[queue_type][queue_id].append((selected_name,))
                assignments_made += 1
                
                logger.info("  %s depth %d: %s (%.1f cm, ref=%.1f)",
                           queue_id, depth, selected_name,
                           best['Alçada (cm)'].iloc[0], ref_height)
            else:
                all_assignments[queue_type][queue_id].append((None,))
        
        if assignments_made == 0:
            logger.info("No assignments made at depth %d - stopping", depth)
            break
        
        # Check balance
        depths_filled = [
            len([d for d in all_assignments[queue_type][qid] if d and d[0]])
            for qid in queue_specs.keys()
        ]
        
        if depths_filled:
            variance = max(depths_filled) - min(depths_filled)
            if variance > 1:
                logger.warning("Queue depth imbalance: variance=%d", variance)
        
        depth += 1
    
    # Summary
    logger.info("\n%s Summary:", queue_type.upper())
    for queue_id in queue_specs.keys():
        filled = len([d for d in all_assignments[queue_type][queue_id] if d and d[0]])
        total = len(all_assignments[queue_type][queue_id])
        logger.info("  %s: %d filled / %d total", queue_id, filled, total)
    
    return all_assignments[queue_type]

def assign_rows_pipeline(
    castellers: pd.DataFrame,
    columns: Dict[str, float],
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    all_assignments: Dict[str, Dict],
    mans_rows: int = 3,
    include_laterals: bool = True,
    include_daus: bool = True
):
    """Pipeline for assigning queue-based positions.
    
    Order: mans → daus → laterals (with remaining castellers)
    """
    result = {}
    stats = {}
    
    # 1. Mans queues
    result['mans'] = assign_queue_positions(
        queue_type='mans',
        queue_specs=MANS_QUEUE_SPECS,
        max_depth=mans_rows,
        castellers=castellers,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
    )
    
    remaining = filter_available_castellers(castellers, all_assignments)
    stats['remaining_after_mans'] = {'total': len(remaining)}
    
    # 2. Daus queues (match mans depth)
    if include_daus:
        result['daus'] = assign_queue_positions(
            queue_type='daus',
            queue_specs=DAUS_QUEUE_SPECS,
            max_depth=mans_rows,
            castellers=castellers,
            column_tronc_heights=column_tronc_heights,
            all_assignments=all_assignments,
        )
        
        remaining = filter_available_castellers(castellers, all_assignments)
        stats['remaining_after_daus'] = {'total': len(remaining)}
    
    # 3. Laterals (use all remaining)
    if include_laterals:
        remaining = filter_available_castellers(castellers, all_assignments)
        auto_depth = max(1, len(remaining) // len(LATERALS_QUEUE_SPECS))
        
        result['laterals'] = assign_queue_positions(
            queue_type='laterals',
            queue_specs=LATERALS_QUEUE_SPECS,
            max_depth=auto_depth,
            castellers=castellers,
            column_tronc_heights=column_tronc_heights,
            all_assignments=all_assignments,
        )
    
    return result, stats