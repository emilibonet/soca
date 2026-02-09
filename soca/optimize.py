import numpy as np
import pandas as pd
import math
import random
from typing import List, Dict, Tuple, Optional, Union
from .data import (
    PositionRequirements,
    OptimizationObjective,
    WeightPreference,
    QueueSpec
)
from .utils import (
    get_logger,
    _get_all_assigned_castellers
)

logger = get_logger(__name__)

MAX_QUEUE_DEPTH = 20

def find_optimal_assignment(
    castellers: pd.DataFrame,
    position_spec: PositionRequirements,
    previous_assignments: Dict[str, Dict],
    columns: Dict[str, int],
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]] = None,
    optimization_method: str = 'exhaustive',
    use_weight: bool = False,
    return_stats: bool = False
) -> Dict[str, Tuple[str, ...]]:
    """
    Unified position assignment algorithm.
    
    Parameters:
        castellers: DataFrame with casteller data
        position_spec: Position requirements specification
        previous_assignments: Dict of all previous position assignments
        columns: Dict mapping column names to base heights
        column_tronc_heights: Optional dict with tronc member heights per column
        optimization_method: 'exhaustive', 'greedy', or 'simulated_annealing'
        use_weight: Whether to use weight data if available
        
    Returns:
        Dict mapping column names to tuples of assigned casteller names
    """
    # Funnel diagnostics - track the filtering process
    total_in_db = len(castellers)
    already_assigned = len(_get_all_assigned_castellers(previous_assignments))

    # Calculate requested slots (needed early for relaxation trigger and shortage factor)
    requested = len(columns) * position_spec.count_per_column

    # Get candidates with expertise (strict filtering)
    candidates_expertise = _get_valid_candidates(castellers, position_spec, allow_relaxed=False, requested_count=requested)
    expertise_count = len(candidates_expertise)
    
    # Print assignment header
    logger.info("Assigning %s", position_spec.position_name.upper())
    
    # Filter out already assigned castellers
    all_assigned = _get_all_assigned_castellers(previous_assignments)
    candidates = candidates_expertise[~candidates_expertise['Nom complet'].isin(all_assigned)]
    available_count = len(candidates)
    
    # Print funnel view
    logger.info("  %s | total=%d | unassigned=%d | expert=%d | avail=%d",
            position_spec.position_name.upper(), total_in_db, total_in_db - already_assigned,
            expertise_count, available_count)
    logger.info("  Requested: %d/col × %d = %d", position_spec.count_per_column, len(columns), requested)
    
    # Check for shortages
    if available_count < requested:
        shortage = requested - available_count
        logger.warning("SHORTAGE: %d slots will remain empty", shortage)
    
    # If no candidates with expertise, try relaxed filtering
    if available_count == 0:
        candidates = _get_valid_candidates(castellers, position_spec, allow_relaxed=True, requested_count=requested)
        candidates = candidates[~candidates['Nom complet'].isin(all_assigned)]
        relaxed_count = len(candidates)
        if relaxed_count > 0:
            logger.info("Using relaxed expertise filtering: %d candidates", relaxed_count)
            available_count = relaxed_count
    
    # If still no candidates, return empty assignment
    if len(candidates) == 0:
        logger.warning("No available candidates for %s", position_spec.position_name)
        return {col_name: () for col_name in columns.keys()}

    # Shortage penalty factor — reduces quality penalties under scarcity (manual §2.2)
    if available_count >= requested:
        shortage_factor = 1.0
    elif available_count >= requested * 0.5:
        shortage_factor = 0.5   # Moderate shortage: 50% penalty reduction
    else:
        shortage_factor = 0.25  # Severe shortage: 75% penalty reduction

    # 3. Calculate reference heights for each column
    reference_heights = _calculate_reference_heights(
        columns, 
        position_spec.reference_positions,
        previous_assignments,
        column_tronc_heights,
        castellers
    )
    
    # 4. Run optimization
    if optimization_method == 'exhaustive':
        assignment, stats = _exhaustive_assignment(
            candidates, columns, reference_heights, position_spec, 
            castellers, use_weight, shortage_factor
        )
    elif optimization_method == 'greedy':
        assignment, stats = _greedy_assignment(
            candidates, columns, reference_heights, position_spec,
            castellers, use_weight, shortage_factor
        )
    elif optimization_method == 'simulated_annealing':
        assignment, stats = simulated_annealing_assignment(
            candidates, columns, reference_heights, position_spec,
            castellers, use_weight, shortage_factor
        )
    elif optimization_method == 'adaptive_simulated_annealing':
        assignment, stats = adaptive_simulated_annealing_assignment(
            candidates, columns, reference_heights, position_spec,
            castellers, use_weight, shortage_factor
        )
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    logger.info("")  # Blank line after optimization

    if return_stats:
        optimization_stats = {'method': optimization_method, **stats}
        optimization_stats['shortage'] = shortage if available_count < requested else 0
        optimization_stats['shortage_pct'] = (shortage / requested * 100) if available_count < requested else 0
        return assignment, optimization_stats
    return assignment


def _get_valid_candidates(
    castellers: pd.DataFrame, 
    position_spec: PositionRequirements,
    allow_relaxed: bool = False,
    requested_count: int = 0,
    return_expertise_counts: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int, int]]:
    """Filter candidates by expertise.
    
    Parameters
    ----------
    castellers : pd.DataFrame
        Full casteller database
    position_spec : PositionRequirements
        Position requirements
    allow_relaxed : bool
        If True and strict filtering yields no candidates, relax expertise requirements
        to include anyone not already assigned
    
    Returns
    -------
    pd.DataFrame
        Filtered candidates
    """
    
    # Determine filtering strictness based on position type
    peripheral_positions = [] # ['daus', 'laterals', 'primeres_mans', 'mans']
    is_peripheral = any(pos in position_spec.position_name for pos in peripheral_positions)
    
    # Start with flexible filtering for peripheral positions
    if is_peripheral:
        allow_relaxed = True
    
    # Check if any expertise keyword matches
    mask = pd.Series(False, index=castellers.index, dtype=bool)
    primary_mask = pd.Series(False, index=castellers.index, dtype=bool)
    secondary_mask = pd.Series(False, index=castellers.index, dtype=bool)
    
    for keyword in position_spec.expertise_keywords:
        pos1_match = castellers['Posició 1'].str.contains(keyword, case=False, na=False)
        pos2_match = castellers['Posició 2'].str.contains(keyword, case=False, na=False)
        
        primary_mask = primary_mask | pos1_match
        secondary_mask = secondary_mask | (pos2_match & ~pos1_match)
        mask = mask | pos1_match | pos2_match
    
    primary_count = primary_mask.sum()
    secondary_count = secondary_mask.sum()

    candidates = castellers[mask].copy()
    
    # For peripheral positions, if few candidates, use more relaxed filtering
    if is_peripheral and requested_count > 0 and len(candidates) < requested_count * 0.5:
        logger.info("Using relaxed filtering for %s (%d specialists < 50%% of %d requested)",
                position_spec.position_name, len(candidates), requested_count)
        # Include candidates with any relevant position experience
        general_keywords = ['primeres', 'Primeres', 'lateral', 'Lateral', 'Dau/Vent', 'dau/vent', 'mans', 'general']
        general_mask = pd.Series(False, index=castellers.index, dtype=bool)
        
        for keyword in general_keywords:
            general_mask = general_mask | (
                castellers['Posició 1'].str.contains(keyword, case=False, na=False) |
                castellers['Posició 2'].str.contains(keyword, case=False, na=False)
            )
        
        general_candidates = castellers[general_mask & ~mask].copy()
        candidates = pd.concat([candidates, general_candidates], ignore_index=True)
    
    # Fix 2: Relaxed expertise for critical shortages (50% threshold)
    critical_positions = ['agulla', 'baix', 'crossa', 'contrafort', 'primeres_mans']
    is_critical = any(pos in position_spec.position_name.lower() for pos in critical_positions)
    
    # Calculate minimum required for critical positions (assume 3 columns standard)
    min_required = position_spec.count_per_column * 3  # Assuming 3 columns
    
    # Apply 50% threshold rule
    if is_critical and len(candidates) < (min_required * 0.5) and allow_relaxed:
        shortage_percent = (len(candidates) / min_required) * 100 if min_required > 0 else 0
        logger.warning("CRITICAL shortage for %s: available=%d (%.0f%%) vs required=%d — relaxing expertise requirements",
                   position_spec.position_name, len(candidates), shortage_percent, min_required)
        candidates = castellers.copy()
    
    # If still no candidates and relaxed mode allowed, return all available castellers
    if len(candidates) == 0 and allow_relaxed:
        logger.warning("No candidates with expertise %s — using all available castellers",
                   position_spec.expertise_keywords)
        candidates = castellers.copy()
    
    # Note: We DON'T filter by assignat flag here - that's handled by the pipeline
    # The all_assignments parameter tracks who's already assigned
    
    # Ensure we have a DataFrame
    if not isinstance(candidates, pd.DataFrame):
        candidates = pd.DataFrame(candidates)
    
    # Add normalized name
    if len(candidates) > 0:
        candidates['name_normalized'] = (
            candidates['Nom complet']
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.lower()
        )
    
    if return_expertise_counts:
        return candidates, primary_count, secondary_count
    
    return candidates

def _calculate_reference_heights(
    columns: Dict[str, int],
    reference_positions: List[str],
    previous_assignments: Dict[str, Dict],
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]],
    castellers: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate reference height for each column.
    
    For baix: returns column base heights (no ratio needed)
    For others: returns combined height of reference positions
    """
    
    reference_heights = {}
    
    if not reference_positions:
        # No reference (e.g., baix) - return base heights
        return columns
    
    for col_name in columns.keys():
        total_height = 0
        
        for ref_pos in reference_positions:
            height_found = False
            
            # First check if it's in previous_assignments (pinya positions)
            if ref_pos in previous_assignments:
                assigned = previous_assignments[ref_pos].get(col_name)
                if isinstance(assigned, (tuple, list)) and assigned:
                    assigned = assigned[0]
                
                if assigned:
                    casteller_data = castellers[castellers['Nom complet'] == assigned]
                    if not casteller_data.empty:
                        # Try multiple column name variations
                        for height_col in ['Alçada (cm)', 'alçada', 'alcada', 'height']:
                            if height_col in casteller_data.columns:
                                height_val = casteller_data[height_col].iloc[0]
                                if pd.notna(height_val):
                                    total_height += height_val
                                    height_found = True
                                    break
            
            # If not found in previous_assignments, check column_tronc_heights (tronc positions)
            if not height_found and column_tronc_heights and col_name in column_tronc_heights:
                if ref_pos in column_tronc_heights[col_name]:
                    total_height += column_tronc_heights[col_name][ref_pos]
                    height_found = True
            
            # If still not found, log a warning
            if not height_found:
                logger.warning("Could not find height for reference position '%s' in column '%s'", ref_pos, col_name)
        
        reference_heights[col_name] = total_height if total_height > 0 else 175.0  # Fallback
    
    return reference_heights

def _should_use_balance_optimization(position_spec: PositionRequirements) -> bool:
    """
    Determine if position needs column balancing for better distribution.
    
    Returns True for positions that should distribute evenly across columns.
    """
    balance_positions = ['daus', 'laterals', 'primeres_mans', 'crossa', 'contrafort']
    return any(pos in position_spec.position_name for pos in balance_positions)


def _calculate_candidate_score(
    candidate: pd.Series,
    target_height: float,
    position_spec: PositionRequirements,
    group_candidates: List[pd.Series] = None,
    use_weight: bool = True,
    depth: Optional[int] = None
) -> float:
    """
    Calculate quality score for a candidate. Lower is better.
    
    Parameters
    ----------
    candidate : pd.Series
        Candidate casteller data
    target_height : float
        Reference height for ratio calculation
    position_spec : PositionRequirements
        Position requirements specification
    group_candidates : List[pd.Series], optional
        Other candidates in the same group (for similarity scoring)
    use_weight : bool
        Whether to use weight data if available
    depth : int, optional
        Queue depth (1-indexed). If == 1, applies strong penalties for
        height/expertise violations in queue positions.
    
    Returns
    -------
    float
        Penalty score (lower is better, 0 is perfect)
    """
    
    score = 0
    height = candidate['Alçada (cm)']
    
    # Detect queue position
    is_queue_position = any(q in position_spec.position_name.lower() 
                           for q in ['mans', 'daus', 'laterals'])
    
    # 1. Height compliance
    min_target = target_height * position_spec.height_ratio_min
    max_target = target_height * position_spec.height_ratio_max
    
    if min_target <= height <= max_target:
        height_score = 0.0
    else:
        if height < min_target:
            height_score = (min_target - height) * position_spec.height_penalty_factor
        else:
            height_score = (height - max_target) * position_spec.height_penalty_factor
        
        # 10x penalty for depth=1 queue height violations
        if depth == 1 and is_queue_position:
            height_score *= 10.0
        
    score += height_score * position_spec.height_weight
    
    # 2. Expertise quality
    pos1 = str(candidate.get('Posició 1', '')).lower()
    pos2 = str(candidate.get('Posició 2', '')).lower()
    
    has_primary = any(kw.lower() in pos1 for kw in position_spec.expertise_keywords)
    has_secondary = any(kw.lower() in pos2 for kw in position_spec.expertise_keywords)
    
    if has_primary:
        expertise_score = 0.0
    elif has_secondary:
        expertise_score = 0.1
    else:
        expertise_score = 0.2
        # CRITICAL: Depth=1 queue positions MUST have primary expertise
        # Catastrophic penalty makes this a near-hard constraint
        if depth == 1 and is_queue_position:
            expertise_score = 50.0  # 250x normal penalty
    
    score += expertise_score * position_spec.expertise_weight
    
    # 3. Weight preference (if data available)
    if use_weight and pd.notna(candidate.get('Pes (kg)')):
        weight = candidate['Pes (kg)']
        
        if position_spec.weight_preference == WeightPreference.HEAVIER:
            # Prefer heavier - penalize lighter
            avg_weight = 70  # Approximate average
            weight_score = (avg_weight - weight) * position_spec.weight_factor
        elif position_spec.weight_preference == WeightPreference.LIGHTER:
            # Prefer lighter - penalize heavier
            avg_weight = 70
            weight_score = (weight - avg_weight) * position_spec.weight_factor
        else:
            weight_score = 0
        
        score += weight_score
    
    # 4. Group similarity (if part of a group)
    if group_candidates:
        valid_group = [c for c in group_candidates if c is not None]
        if valid_group:
            all_heights = [c['Alçada (cm)'] for c in valid_group] + [height]
            height_variance = np.var(all_heights)
            score += height_variance * position_spec.height_similarity_weight * 0.1
    
    return score


def _greedy_assignment(
    candidates: pd.DataFrame,
    columns: Dict[str, int],
    reference_heights: Dict[str, float],
    position_spec: PositionRequirements,
    castellers: pd.DataFrame,
    use_weight: bool,
    shortage_factor: float = 1.0,
    return_stats: bool = True
) -> Tuple[Dict[str, Tuple[str, ...]], Dict]:
    """Greedy assignment algorithm - works for all position types."""
    stats = {'iterations': 0, 'final_score': 0, 'initial_score': 0}
    
    assignment = {}
    used = set()
    should_balance = _should_use_balance_optimization(position_spec)
    
    # Sort columns by constraint (for baix: base height; for others: reference height)
    if position_spec.optimization_objective == OptimizationObjective.COLUMN_BALANCE:
        # For baix: sort by base height descending
        sorted_columns = sorted(columns.items(), key=lambda x: x[1], reverse=True)
    else:
        # For others: sort by reference height
        sorted_columns = sorted(
            reference_heights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
    
    # For balance optimization, use round-robin approach
    if should_balance:
        # Calculate target distribution
        total_slots = len(columns) * position_spec.count_per_column
        available_candidates = len(candidates)
        
        if available_candidates < total_slots:
            # Adjust for available candidates
            target_per_column = max(1, available_candidates // len(columns))
        else:
            target_per_column = position_spec.count_per_column
        
        # Use round-robin assignment to ensure balance
        column_cycle = list(columns.keys())
        current_col_idx = 0
        assignments_made = 0
        
        used = set()
        consecutive_skips = 0  # ← ADD THIS
        
        while assignments_made < min(total_slots, available_candidates):
            col_name = column_cycle[current_col_idx]
            
            current_assigned = len(assignment.get(col_name, ()))
            if current_assigned >= target_per_column:
                current_col_idx = (current_col_idx + 1) % len(column_cycle)
                consecutive_skips += 1  # ← ADD THIS
                
                if consecutive_skips >= len(column_cycle):
                    # All columns full, stop
                    break
                continue
            
            consecutive_skips = 0  # ← RESET on progress
            
            # Find best available candidate
            available = candidates[~candidates['Nom complet'].isin(used)]
            if available.empty:
                break
            
            ref_height = reference_heights[col_name]
            selected_in_col = assignment.get(col_name, ())
            
            best_candidate = None
            best_score = float('inf')
            
            for _, candidate in available.iterrows():
                score = _calculate_candidate_score(
                    candidate,
                    ref_height,
                    position_spec,
                    [candidates[candidates['Nom complet'] == s].iloc[0] if s in candidates['Nom complet'].values else None 
                     for s in selected_in_col],
                    use_weight
                ) * shortage_factor
                
                if score < best_score:
                    best_score = score
                    best_candidate = candidate['Nom complet']
            
            if best_candidate:
                assignment[col_name] = assignment.get(col_name, ()) + (best_candidate,)
                used.add(best_candidate)
                assignments_made += 1
            
            current_col_idx = (current_col_idx + 1) % len(column_cycle)
    
    else:
        # Original greedy approach for non-balanced positions
        for col_name, ref_height in sorted_columns:
            if col_name not in columns:
                continue
                
            selected = []
            
            for _ in range(position_spec.count_per_column):
                # Find best candidate for this slot
                available = candidates[~candidates['Nom complet'].isin(used)]
                
                if available.empty:
                    break
                
                best_candidate = None
                best_score = float('inf')
                
                for _, candidate in available.iterrows():
                    score = _calculate_candidate_score(
                        candidate,
                        ref_height,
                        position_spec,
                        [candidates[candidates['Nom complet'] == s].iloc[0] for s in selected],
                        use_weight
                    ) * shortage_factor
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate['Nom complet']
                
                if best_candidate:
                    selected.append(best_candidate)
                    used.add(best_candidate)
            
            assignment[col_name] = tuple(selected) if selected else ()
    
    return assignment, stats


def _exhaustive_assignment(
    candidates: pd.DataFrame,
    columns: Dict[str, int],
    reference_heights: Dict[str, float],
    position_spec: PositionRequirements,
    castellers: pd.DataFrame,
    use_weight: bool,
    shortage_factor: float = 1.0,
    return_stats: bool = True
) -> Tuple[Dict[str, Tuple[str, ...]], Dict]:
    """Exhaustive search - adapts to position type."""
    
    from itertools import combinations, product

    stats = {'iterations': 0, 'final_score': 0, 'initial_score': 0}
    
    column_names = list(columns.keys())
    candidate_names = list(candidates['Nom complet'])
    
    # Check feasibility
    total_needed = len(column_names) * position_spec.count_per_column
    if len(candidate_names) < total_needed:
        logger.warning("Not enough candidates; falling back to greedy")
        return _greedy_assignment(
            candidates, columns, reference_heights, position_spec, castellers, use_weight, shortage_factor
        )
    
    # Generate combinations per column
    all_column_combos = {}
    for col_name in column_names:
        if position_spec.count_per_column == 1:
            combos = [(name,) for name in candidate_names]
        else:
            combos = list(combinations(candidate_names, position_spec.count_per_column))
        all_column_combos[col_name] = combos
    
    # Check if search space is too large
    total_combos = 1
    for combos in all_column_combos.values():
        total_combos *= len(combos)
    
    if total_combos > 1000000:
        logger.warning("Search space too large (%d combos); falling back to greedy", total_combos)
        return _greedy_assignment(
            candidates, columns, reference_heights, position_spec, castellers, use_weight, shortage_factor
        )
    
    logger.info("Evaluating %d possible assignments...", total_combos)
    
    # Search all assignments
    best_assignment = None
    best_score = float('inf')
    
    for assignment in product(*[all_column_combos[col] for col in column_names]):
        # Check for conflicts
        all_assigned = [c for col_assignment in assignment for c in col_assignment]
        if len(set(all_assigned)) != len(all_assigned):
            continue
        
        # Score this assignment
        score = _score_complete_assignment(
            dict(zip(column_names, assignment)),
            reference_heights,
            position_spec,
            castellers,
            columns,
            use_weight,
            shortage_factor
        )
        
        if score < best_score:
            best_score = score
            best_assignment = dict(zip(column_names, assignment))
    
    return (best_assignment if best_assignment else {}), stats

def _score_complete_assignment(
    assignment: Dict[str, Tuple[str, ...]],
    reference_heights: Dict[str, float],
    position_spec: PositionRequirements,
    castellers: pd.DataFrame,
    columns: Dict[str, int],
    use_weight: bool,
    shortage_factor: float = 1.0,
) -> float:
    """Score a complete assignment."""
    
    total_score = 0
    
    if position_spec.optimization_objective == OptimizationObjective.COLUMN_BALANCE:
        # For baix: optimize column height balance (structural — not scaled by shortage_factor)
        column_totals = []
        for col_name, assigned in assignment.items():
            baix_height = sum(
                castellers[castellers['Nom complet'] == name]['Alçada (cm)'].iloc[0]
                for name in assigned
            )
            column_totals.append(columns[col_name] + baix_height)
        
        # Minimize variance
        total_score = np.var(column_totals)
        
        # Add weight balance if available
        if use_weight:
            column_weights = []
            for col_name, assigned in assignment.items():
                weights = [
                    castellers[castellers['Nom complet'] == name]['Pes (kg)'].iloc[0]
                    for name in assigned
                    if pd.notna(castellers[castellers['Nom complet'] == name]['Pes (kg)'].iloc[0])
                ]
                if weights:
                    column_weights.append(sum(weights))
            
            if len(column_weights) > 1:
                total_score += np.var(column_weights) * 0.3

    elif position_spec.optimization_objective == OptimizationObjective.FILL_ALL_REQUIRED:
        # Agulles: heavy fill penalty (structural, not scaled) + quality (scaled by shortage_factor)
        for col_name in columns.keys():
            assigned = assignment.get(col_name, ())
            filled = [n for n in assigned if n]
            missing = position_spec.count_per_column - len(filled)
            if missing > 0:
                total_score += missing * 10.0  # Fill priority per manual §2.4
            for name in filled:
                candidate = castellers[castellers['Nom complet'] == name].iloc[0]
                total_score += _calculate_candidate_score(
                    candidate,
                    reference_heights.get(col_name, 175.0),
                    position_spec,
                    [castellers[castellers['Nom complet'] == n].iloc[0] for n in filled if n != name],
                    use_weight
                ) * shortage_factor

    elif position_spec.optimization_objective == OptimizationObjective.EVEN_DISTRIBUTION:
        # Crosses, contraforts: distribution variance penalty (structural, not scaled) + quality (scaled)
        counts = [len(assignment.get(col, ())) for col in columns.keys()]
        total_score += np.var(counts) * 3.0  # Distribution weight per manual §2.4
        for col_name, assigned in assignment.items():
            ref_height = reference_heights[col_name]
            for name in assigned:
                candidate = castellers[castellers['Nom complet'] == name].iloc[0]
                total_score += _calculate_candidate_score(
                    candidate,
                    ref_height,
                    position_spec,
                    [castellers[castellers['Nom complet'] == n].iloc[0] for n in assigned if n != name],
                    use_weight
                ) * shortage_factor

    else:
        # HEIGHT_COMPLIANCE: per-candidate quality scoring (scaled by shortage_factor)
        for col_name, assigned in assignment.items():
            ref_height = reference_heights[col_name]
            
            for name in assigned:
                candidate = castellers[castellers['Nom complet'] == name].iloc[0]
                total_score += _calculate_candidate_score(
                    candidate,
                    ref_height,
                    position_spec,
                    [castellers[castellers['Nom complet'] == n].iloc[0] for n in assigned if n != name],
                    use_weight
                ) * shortage_factor
    
    return total_score




def simulated_annealing_assignment(
    candidates: pd.DataFrame,
    columns: Dict[str, int],
    reference_heights: Dict[str, float],
    position_spec: PositionRequirements,
    castellers: pd.DataFrame,
    use_weight: bool,
    shortage_factor: float = 1.0,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    iterations_per_temp: int = 100,
    min_temp: float = 0.1,
    return_stats: bool = True
) -> Tuple[Dict[str, Tuple[str, ...]], Dict]:
    """
    Simulated annealing assignment algorithm.
    
    Balances exploration and exploitation to find high-quality assignments
    while avoiding local optima. Good for medium-sized search spaces where
    exhaustive is too slow but greedy might miss better solutions.
    
    Parameters:
        candidates: Available casteller candidates
        columns: Column definitions
        reference_heights: Reference heights per column
        position_spec: Position requirements
        castellers: Full casteller database
        use_weight: Whether to use weight data
        initial_temp: Starting temperature for annealing
        cooling_rate: Temperature reduction factor (0.9-0.99)
        iterations_per_temp: Iterations at each temperature
        min_temp: Stop when temperature reaches this value
        
    Returns:
        Optimal assignment found
    """
    
    column_names = list(columns.keys())
    candidate_names = list(candidates['Nom complet'])
    
    # Check feasibility
    total_needed = len(column_names) * position_spec.count_per_column
    if len(candidate_names) < total_needed:
        logger.warning("Not enough candidates; falling back to greedy")
        return _greedy_assignment(
            candidates, columns, reference_heights, position_spec, castellers, use_weight, shortage_factor
        )
    
    # Generate initial solution (random valid assignment)
    current_solution = _generate_random_valid_assignment(
        candidate_names, column_names, position_spec.count_per_column
    )
    current_score = _score_complete_assignment(
        current_solution, reference_heights, position_spec, 
        castellers, columns, use_weight, shortage_factor
    )
    
    # Track best solution found
    best_solution = current_solution.copy()
    best_score = current_score
    
    # Track stats
    stats = {
        'initial_score': current_score,
        'progress': [],
        'iterations': 0
    }
    
    temperature = initial_temp
    iteration = 0
    total_iterations = 0
    
    logger.info("Starting simulated annealing (initial score: %.2f)", current_score)
    
    # Annealing loop
    while temperature > min_temp:
        for _ in range(iterations_per_temp):
            total_iterations += 1
            
            # Generate neighbor solution
            neighbor = _generate_neighbor_assignment(
                current_solution, column_names, candidate_names, position_spec.count_per_column
            )
            
            # Score neighbor
            neighbor_score = _score_complete_assignment(
                neighbor, reference_heights, position_spec,
                castellers, columns, use_weight, shortage_factor
            )
            
            # Decide whether to accept neighbor
            delta = neighbor_score - current_score
            
            if delta < 0:
                # Better solution - always accept
                current_solution = neighbor
                current_score = neighbor_score
                
                # Update best if necessary
                if current_score < best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
            else:
                # Worse solution - accept probabilistically
                acceptance_probability = math.exp(-delta / temperature)
                
                if random.random() < acceptance_probability:
                    current_solution = neighbor
                    current_score = neighbor_score
        
        # Cool down
        iteration += 1
        temperature *= cooling_rate
        
        # Progress report only every 1000 total iterations
        if total_iterations % 1000 == 0:
            logger.info("Progress: %d iterations, best=%.2f", total_iterations, best_score)
            stats['progress'].append((total_iterations, best_score))
    
    stats['iterations'] = total_iterations
    stats['final_score'] = best_score
    
    logger.info("Completed in %d iterations (score: %.2f)", total_iterations, best_score)
    
    return best_solution, stats


def _generate_random_valid_assignment(
    candidate_names: List[str],
    column_names: List[str],
    count_per_column: int
) -> Dict[str, Tuple[str, ...]]:
    """
    Generate a random valid assignment (no conflicts).
    
    Used as initial solution for simulated annealing.
    """
    assignment = {}
    available = candidate_names.copy()
    random.shuffle(available)
    
    idx = 0
    for col_name in column_names:
        selected = []
        for _ in range(count_per_column):
            if idx < len(available):
                selected.append(available[idx])
                idx += 1
        assignment[col_name] = tuple(selected)
    
    return assignment


def _generate_neighbor_assignment(
    current_assignment: Dict[str, Tuple[str, ...]],
    column_names: List[str],
    all_candidates: List[str],
    count_per_column: int
) -> Dict[str, Tuple[str, ...]]:
    """
    Generate a neighbor solution by making a small change to current assignment.
    
    Possible moves:
    1. Swap two castellers between different columns
    2. Swap a casteller within a column (if count_per_column > 1)
    3. Replace a casteller with an unassigned candidate
    
    Returns a new assignment with one of these modifications.
    """
    neighbor = {col: list(assigned) for col, assigned in current_assignment.items()}
    
    # Get all currently assigned castellers
    all_assigned = set()
    for assigned_list in neighbor.values():
        all_assigned.update(c for c in assigned_list if c is not None)
    
    # Get unassigned candidates
    unassigned = [c for c in all_candidates if c not in all_assigned]
    
    # Choose a random move type
    move_types = ['swap_between_columns', 'swap_within_column', 'replace_with_unassigned']
    weights = [0.5, 0.2, 0.3]  # Prefer swaps between columns
    
    if count_per_column == 1:
        # Can't swap within column if only 1 per column
        move_types.remove('swap_within_column')
        weights = [0.6, 0.4]
    
    if not unassigned:
        # No unassigned candidates
        move_types.remove('replace_with_unassigned')
        weights = [0.7, 0.3] if count_per_column > 1 else [1.0]
    
    move_type = random.choices(move_types, weights=weights)[0]
    
    if move_type == 'swap_between_columns':
        # Swap castellers between two different columns
        col1, col2 = random.sample(column_names, 2)
        
        if neighbor[col1] and neighbor[col2]:
            idx1 = random.randint(0, len(neighbor[col1]) - 1)
            idx2 = random.randint(0, len(neighbor[col2]) - 1)
            
            # Swap
            neighbor[col1][idx1], neighbor[col2][idx2] = neighbor[col2][idx2], neighbor[col1][idx1]
    
    elif move_type == 'swap_within_column':
        # Swap two castellers within the same column
        col = random.choice(column_names)
        
        if len(neighbor[col]) >= 2:
            idx1, idx2 = random.sample(range(len(neighbor[col])), 2)
            neighbor[col][idx1], neighbor[col][idx2] = neighbor[col][idx2], neighbor[col][idx1]
    
    elif move_type == 'replace_with_unassigned':
        # Replace a random assigned casteller with an unassigned one
        col = random.choice(column_names)
        
        if neighbor[col] and unassigned:
            idx = random.randint(0, len(neighbor[col]) - 1)
            replacement = random.choice(unassigned)
            neighbor[col][idx] = replacement
    
    # Convert back to tuples
    return {col: tuple(assigned) for col, assigned in neighbor.items()}


def adaptive_simulated_annealing_assignment(
    candidates: pd.DataFrame,
    columns: Dict[str, int],
    reference_heights: Dict[str, float],
    position_spec: PositionRequirements,
    castellers: pd.DataFrame,
    use_weight: bool,
    shortage_factor: float = 1.0,
    max_iterations: int = 5000,
    return_stats: bool = True
) -> Tuple[Dict[str, Tuple[str, ...]], Dict]:
    """
    Simulated annealing with adaptive cooling schedule.
    
    Adjusts cooling rate based on acceptance rate - if we're accepting
    too many bad solutions, cool faster; if we're stuck, cool slower.
    """
    column_names = list(columns.keys())
    candidate_names = list(candidates['Nom complet'])
    
    total_needed = len(column_names) * position_spec.count_per_column
    if len(candidate_names) < total_needed:
        logger.warning("Not enough candidates; falling back to greedy")
        return _greedy_assignment(
            candidates, columns, reference_heights, position_spec, castellers, use_weight, shortage_factor
        )
    
    # Generate initial solution
    current_solution = _generate_random_valid_assignment(
        candidate_names, column_names, position_spec.count_per_column
    )
    current_score = _score_complete_assignment(
        current_solution, reference_heights, position_spec,
        castellers, columns, use_weight, shortage_factor
    )
    
    best_solution = current_solution.copy()
    best_score = current_score
    
    # Track stats
    stats = {
        'initial_score': current_score,
        'progress': [],
        'iterations': 0
    }
    
    # Adaptive temperature initialization
    # Run sample evaluations to estimate score variance
    sample_scores = []
    for _ in range(20):
        sample = _generate_random_valid_assignment(
            candidate_names, column_names, position_spec.count_per_column
        )
        sample_scores.append(_score_complete_assignment(
            sample, reference_heights, position_spec, castellers, columns, use_weight, shortage_factor
        ))
    
    # Set initial temperature to ~2x standard deviation of scores
    temperature = np.std(sample_scores) * 2
    
    logger.info("Starting adaptive annealing (initial score: %.2f)", current_score)
    
    iteration = 0
    no_improvement_count = 0
    acceptance_count = 0
    evaluation_window = 100
    
    while iteration < max_iterations:
        iteration += 1
        
        # Generate and evaluate neighbor
        neighbor = _generate_neighbor_assignment(
            current_solution, column_names, candidate_names, position_spec.count_per_column
        )
        neighbor_score = _score_complete_assignment(
            neighbor, reference_heights, position_spec,
            castellers, columns, use_weight, shortage_factor
        )
        
        # Acceptance decision
        delta = neighbor_score - current_score
        
        if delta < 0:
            # Better solution
            current_solution = neighbor
            current_score = neighbor_score
            acceptance_count += 1
            no_improvement_count = 0
            
            if current_score < best_score:
                best_solution = current_solution.copy()
                best_score = current_score
                # Removed verbose best iteration logging
        else:
            # Worse solution - probabilistic acceptance
            acceptance_prob = math.exp(-delta / temperature)
            
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_score = neighbor_score
                acceptance_count += 1
            
            no_improvement_count += 1
        
        # Adaptive cooling every 'evaluation_window' iterations
        if iteration % evaluation_window == 0:
            acceptance_rate = acceptance_count / evaluation_window
            
            # Adjust cooling based on acceptance rate
            # Target: 20-40% acceptance rate
            if acceptance_rate > 0.4:
                # Accepting too much - cool faster
                temperature *= 0.85
            elif acceptance_rate < 0.2:
                # Too picky - cool slower (or heat up slightly)
                temperature *= 0.98
            else:
                # Good range - normal cooling
                temperature *= 0.9
            
            # Only log progress every 1000 iterations
            if iteration % 1000 == 0:
                stats['progress'].append((iteration, best_score))
                logger.info("Progress: %d iterations, best=%.2f", iteration, best_score)
            
            acceptance_count = 0
        
        # Stop if stuck for too long
        if no_improvement_count > 1000:
            stats['stop_reason'] = 'no improvement (1000 iters)'
            logger.info("No improvement for 1000 iterations - stopping early")
            break
        
        # Stop if temperature too low
        if temperature < 0.01:
            stats['stop_reason'] = 'temperature too low'
            logger.info("Temperature too low - stopping")
            break
    
    stats['iterations'] = iteration
    stats['final_score'] = best_score
    
    logger.info("Completed in %d iterations (score: %.2f)", iteration, best_score)
    
    return best_solution, stats

def _calculate_reference_heights_for_queue(
    queue_spec: 'QueueSpec',
    depth: int,
    all_assignments: Dict,
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]],
    castellers: pd.DataFrame
) -> float:
    """Calculate reference height for a queue position at given depth.
    
    FIXES:
    - Depth 1: Use MAX(column heights) instead of average for multi-column queues
    - Depth 1: Properly sum baix+segon per column before taking max
    - Depth >1: No change (uses previous depth in same queue)
    
    Parameters
    ----------
    queue_spec : QueueSpec
        Queue specification
    depth : int
        Depth level (1-indexed)
    all_assignments : dict
        Current assignments (finalized)
    column_tronc_heights : dict
        Tronc heights per column
    castellers : pd.DataFrame
        Casteller database
        
    Returns
    -------
    float
        Reference height in cm
    """
    if depth == 1:
        # Use baix + segon from column(s)
        # FIX: For multi-column queues (daus), use MAX to represent constraining column
        column_heights = []
        
        for col in queue_spec.column_refs:
            col_total = 0
            found_baix = False
            found_segon = False
            
            # Get baix height
            baix_assignment = all_assignments.get('baix', {}).get(col)
            if baix_assignment and len(baix_assignment) > 0 and baix_assignment[0]:
                baix_data = castellers[castellers['Nom complet'] == baix_assignment[0]]
                if not baix_data.empty:
                    col_total += baix_data['Alçada (cm)'].iloc[0]
                    found_baix = True
            
            # Get segon height
            if column_tronc_heights and col in column_tronc_heights:
                segon_height = column_tronc_heights[col].get('segon', 0)
                if segon_height > 0:
                    col_total += segon_height
                    found_segon = True
            
            # Only include columns where we have both baix and segon
            if found_baix and found_segon:
                column_heights.append(col_total)
        
        if not column_heights:
            logger.warning("Could not find baix+segon for %s depth 1 - using fallback 350cm",
                          queue_spec.queue_id)
            return 350.0
        
        # FIX: Use MAX for multi-column (daus must satisfy tallest column constraint)
        return max(column_heights)
    else:
        # Use previous depth in same queue
        queue_type = queue_spec.queue_type
        queue_id = queue_spec.queue_id
        
        if queue_type in all_assignments and queue_id in all_assignments[queue_type]:
            prev_assignments = all_assignments[queue_type][queue_id]
            if len(prev_assignments) >= depth - 1:
                prev_assignment = prev_assignments[depth - 2]
                if prev_assignment and len(prev_assignment) > 0 and prev_assignment[0]:
                    prev_data = castellers[castellers['Nom complet'] == prev_assignment[0]]
                    if not prev_data.empty:
                        return prev_data['Alçada (cm)'].iloc[0]
        
        # Fallback
        logger.warning("Could not find reference for %s %s depth %d - using fallback 175cm",
                      queue_type, queue_id, depth)
        return 175.0

def create_position_spec_from_queue(
    queue_spec: 'QueueSpec',
    depth: int
) -> PositionRequirements:
    """Create a PositionRequirements from a QueueSpec for a specific depth.
    
    Parameters
    ----------
    queue_spec : QueueSpec
        Queue specification
    depth : int
        Depth level (1-indexed)
        
    Returns
    -------
    PositionRequirements
        Position spec for this depth level
    """
    # Depth 1: use spec's ratios; Depth N>1: use depth_gt1 ratios
    if depth == 1:
        height_min = queue_spec.height_ratio_min
        height_max = queue_spec.height_ratio_max
    else:
        height_min = queue_spec.queue_height_ratio_min
        height_max = queue_spec.queue_height_ratio_max
    
    return PositionRequirements(
        position_name=f'{queue_spec.queue_type}_{queue_spec.queue_id}_depth{depth}',
        expertise_keywords=queue_spec.expertise_keywords,
        count_per_column=1,
        reference_positions=[],  # Handled by custom height calculation
        height_ratio_min=height_min,
        height_ratio_max=height_max,
        optimization_objective=OptimizationObjective.HEIGHT_COMPLIANCE,
        weight_preference=queue_spec.weight_preference,
        height_weight=queue_spec.height_weight,
        expertise_weight=queue_spec.expertise_weight,
        height_similarity_weight=queue_spec.height_similarity_weight,
        weight_factor=queue_spec.weight_factor,
        height_penalty_factor=getattr(queue_spec, 'height_penalty_factor', 0.1)
    )

def global_peripheral_optimization(
    mans_specs: Dict[str, QueueSpec],
    daus_specs: Dict[str, QueueSpec],
    laterals_specs: Dict[str, QueueSpec],
    mans_depth: int,
    daus_depth: int,
    laterals_depth: int,
    available_castellers: pd.DataFrame,
    all_castellers: pd.DataFrame,
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    all_assignments: Dict,
    use_weight: bool = True,
    max_iterations: int = 10000
) -> Tuple[Dict[str, Dict[str, List[Tuple[str, ...]]]], Dict]:
    """Global optimization across mans, daus, and laterals simultaneously."""
    # Merge all specs with type prefix
    all_specs = {}
    for qid, spec in mans_specs.items():
        all_specs[f'mans:{qid}'] = (spec, mans_depth, 'mans')
    for qid, spec in daus_specs.items():
        all_specs[f'daus:{qid}'] = (spec, daus_depth, 'daus')
    for qid, spec in laterals_specs.items():
        all_specs[f'laterals:{qid}'] = (spec, laterals_depth, 'laterals')
    
    candidate_names = list(available_castellers['Nom complet'])
    
    # Generate initial solution
    current_solution = _generate_random_global_assignment(all_specs, candidate_names)
    current_score = _score_global_assignment(
        current_solution, all_specs, all_assignments, 
        column_tronc_heights, all_castellers, use_weight
    )
    
    best_solution = current_solution.copy()
    best_score = current_score
    
    # Adaptive temperature
    sample_scores = []
    for _ in range(20):
        sample = _generate_random_global_assignment(all_specs, candidate_names)
        sample_scores.append(_score_global_assignment(
            sample, all_specs, all_assignments, column_tronc_heights, 
            all_castellers, use_weight
        ))
    temperature = np.std(sample_scores) * 2 if len(sample_scores) > 1 else 100.0
    
    logger.info("Starting global peripheral optimization (initial score: %.2f)", current_score)
    
    iteration = 0
    no_improvement = 0
    acceptance_count = 0
    
    while iteration < max_iterations and temperature > 0.01:
        iteration += 1
        
        # Generate neighbor
        neighbor = _generate_neighbor_global_assignment(
            current_solution, all_specs, candidate_names
        )
        neighbor_score = _score_global_assignment(
            neighbor, all_specs, all_assignments, 
            column_tronc_heights, all_castellers, use_weight
        )
        
        delta = neighbor_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution = neighbor
            current_score = neighbor_score
            acceptance_count += 1
            
            if current_score < best_score:
                best_solution = current_solution.copy()
                best_score = current_score
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        
        # Adaptive cooling
        if iteration % 100 == 0:
            acceptance_rate = acceptance_count / 100
            if acceptance_rate > 0.4:
                temperature *= 0.85
            elif acceptance_rate < 0.2:
                temperature *= 0.98
            else:
                temperature *= 0.9
            acceptance_count = 0
            
            if iteration % 1000 == 0:
                logger.info("Iteration %d, best=%.2f", iteration, best_score)
        
        if no_improvement > 1500:
            break
    
    logger.info("Completed in %d iterations (score: %.2f)", iteration, best_score)
    
    # Unpack solution by type
    result = {'mans': {}, 'daus': {}, 'laterals': {}}
    for key, assignments in best_solution.items():
        queue_type, queue_id = key.split(':', 1)
        result[queue_type][queue_id] = assignments
    
    stats = {'final_score': best_score, 'iterations': iteration}
    return result, stats


def _generate_random_global_assignment(
    all_specs: Dict[str, Tuple[QueueSpec, int, str]],
    candidate_names: List[str]
) -> Dict[str, List[Tuple[str, ...]]]:
    """Generate random assignment across all queue types."""
    assignment = {}
    available = candidate_names.copy()
    random.shuffle(available)
    
    idx = 0
    for key, (spec, max_depth, qtype) in all_specs.items():
        assignment[key] = []
        for depth in range(max_depth):
            if idx < len(available):
                assignment[key].append((available[idx],))
                idx += 1
            else:
                assignment[key].append((None,))
    
    return assignment


def _generate_neighbor_global_assignment(
    current: Dict[str, List[Tuple[str, ...]]],
    all_specs: Dict[str, Tuple[QueueSpec, int, str]],
    all_candidates: List[str]
) -> Dict[str, List[Tuple[str, ...]]]:
    """Generate neighbor by swapping or replacing across queues/types."""
    import random
    
    neighbor = {k: [a for a in v] for k, v in current.items()}
    
    # Get assigned and unassigned
    all_assigned = set()
    assigned_positions = []
    for key in all_specs.keys():
        for depth in range(len(neighbor[key])):
            if neighbor[key][depth] and neighbor[key][depth][0]:
                assigned_positions.append((key, depth))
                all_assigned.add(neighbor[key][depth][0])
    
    unassigned = [c for c in all_candidates if c not in all_assigned]
    
    if len(assigned_positions) < 2:
        return neighbor
    
    # Choose move type
    move_types = ['swap', 'replace'] if unassigned else ['swap']
    move = random.choice(move_types)
    
    if move == 'swap':
        # Swap between any two positions
        (key1, depth1), (key2, depth2) = random.sample(assigned_positions, 2)
        neighbor[key1][depth1], neighbor[key2][depth2] = neighbor[key2][depth2], neighbor[key1][depth1]
    else:
        # Replace with unassigned
        key, depth = random.choice(assigned_positions)
        replacement = random.choice(unassigned)
        neighbor[key][depth] = (replacement,)
    
    return neighbor


def _score_global_assignment(
    assignment: Dict[str, List[Tuple[str, ...]]],
    all_specs: Dict[str, Tuple[QueueSpec, int, str]],
    all_assignments: Dict,
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    castellers: pd.DataFrame,
    use_weight: bool
) -> float:
    """Score complete global assignment with strong monotonic penalties."""
    total_score = 0.0
    
    for key, depth_list in assignment.items():
        spec, max_depth, queue_type = all_specs[key]
        prev_height = None
        
        for depth_idx, assignment_tuple in enumerate(depth_list, start=1):
            if not assignment_tuple or not assignment_tuple[0]:
                total_score += 5.0
                continue
            
            name = assignment_tuple[0]
            candidate = castellers[castellers['Nom complet'] == name]
            if candidate.empty:
                total_score += 100.0
                continue
            
            candidate_row = candidate.iloc[0]
            current_height = candidate_row['Alçada (cm)']
            
            # Calculate reference with in-progress context
            ref_height = _calculate_reference_for_global_assignment(
                spec, depth_idx, key, assignment, all_assignments,
                column_tronc_heights, castellers
            )
            
            pos_spec = create_position_spec_from_queue(spec, depth_idx)
            score = _calculate_candidate_score(
                candidate_row, ref_height, pos_spec, [], use_weight,
                depth=depth_idx
            )
            total_score += score
            
            # STRONG monotonic penalty
            if depth_idx > 1 and prev_height is not None:
                if current_height >= prev_height:
                    violation = current_height - prev_height + 1
                    total_score += violation * 20.0  # 10x stronger
            
            prev_height = current_height
    
    # Balance penalty across types
    type_depths = {'mans': [], 'daus': [], 'laterals': []}
    for key, depth_list in assignment.items():
        _, _, queue_type = all_specs[key]
        filled = len([d for d in depth_list if d and d[0]])
        type_depths[queue_type].append(filled)
    
    # Within-type balance
    for qtype, depths in type_depths.items():
        if depths:
            variance = max(depths) - min(depths)
            total_score += variance * 10.0
    
    return total_score


def _calculate_reference_for_global_assignment(
    queue_spec: QueueSpec,
    depth: int,
    current_key: str,
    global_assignment: Dict[str, List[Tuple[str, ...]]],
    all_assignments: Dict,
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    castellers: pd.DataFrame
) -> float:
    """Calculate reference height in global optimization context."""
    if depth == 1:
        # Use baix + segon
        column_heights = []
        for col in queue_spec.column_refs:
            col_total = 0
            found_baix = found_segon = False
            
            baix_assignment = all_assignments.get('baix', {}).get(col)
            if baix_assignment and baix_assignment[0]:
                baix_data = castellers[castellers['Nom complet'] == baix_assignment[0]]
                if not baix_data.empty:
                    col_total += baix_data['Alçada (cm)'].iloc[0]
                    found_baix = True
            
            if column_tronc_heights and col in column_tronc_heights:
                segon_height = column_tronc_heights[col].get('segon', 0)
                if segon_height > 0:
                    col_total += segon_height
                    found_segon = True
            
            if found_baix and found_segon:
                column_heights.append(col_total)
        
        return max(column_heights) if column_heights else 350.0
    else:
        # Use previous depth in same queue
        prev_assignments = global_assignment.get(current_key, [])
        if len(prev_assignments) >= depth - 1:
            prev_assignment = prev_assignments[depth - 2]
            if prev_assignment and prev_assignment[0]:
                prev_data = castellers[castellers['Nom complet'] == prev_assignment[0]]
                if not prev_data.empty:
                    return prev_data['Alçada (cm)'].iloc[0]
        
        return 175.0
