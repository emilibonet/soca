"""
Unified row assignment helpers for SOCA.

Provides:
- assign_sequential_mans: build primeres mans + N sequential mans rows (80-100% stepping)
- assign_laterals: assign lateral positions (51% fixed ratio of baix+segon)
- assign_daus_or_vents: assign daus/vents (gap-fillers, prefer heavier)

This module depends on the core optimizer already implemented in optimize.py
and reuses `find_optimal_assignment`, `POSITION_SPECS`, and the
PositionRequirements dataclass.

Drop this file next to optimize.py and import the helpers from your pipeline.
"""
from typing import Dict, Tuple, Optional
import dataclasses

# Import core utilities from optimize.py
from .optimize import (
    find_optimal_assignment,
    POSITION_SPECS,
    PositionRequirements,
    OptimizationObjective,
    WeightPreference,
)


def assign_sequential_mans(
    castellers,
    columns: Dict[str, int],
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]],
    all_assignments: Dict[str, Dict[str, Tuple[str, ...]]],
    n_rows: int = 3,
    optimization_method: str = 'greedy',
    use_weight: bool = True,
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Assign primeres_mans and the requested number of sequential "mans" rows.

    - The first row uses the existing `POSITION_SPECS['primeres_mans']` spec.
    - Each subsequent row references the previous row and uses the
      80-100% height ratio rule (height_ratio_min=0.8, height_ratio_max=1.0).

    Parameters
    ----------
    castellers: pd.DataFrame
        Casteller database (same schema used by optimize.find_optimal_assignment)
    columns: Dict[str,int]
        Column base heights (tronc totals)
    column_tronc_heights: Optional[Dict]
        Tronc heights (for reference positions like 'segon')
    all_assignments: Dict
        Mutable dict of already-made assignments; will be updated in-place
    n_rows: int
        Total number of mans rows to create (including primeres_mans). If 0 nothing is done.
    optimization_method: str
        'exhaustive', 'greedy', 'simulated_annealing', etc.

    Returns
    -------
    Dict[str, Dict[str, Tuple[str,...]]]
        A mapping of the newly created mans rows to their assignments.
    """
    if n_rows <= 0:
        return {}

    created = {}

    # 1) Primeres mans (row 1)
    primeres_key = 'primeres_mans'
    if primeres_key in POSITION_SPECS:
        spec = POSITION_SPECS[primeres_key]
    else:
        # Fallback spec: prefer taller, heavier
        spec = PositionRequirements(
            position_name=primeres_key,
            expertise_keywords=['primeres', 'primeres_mans', 'general'],
            count_per_column=1,
            reference_positions=['baix', 'segon'],
            height_ratio_min=0.52,
            height_ratio_max=1.0,
            optimization_objective=OptimizationObjective.HEIGHT_COMPLIANCE,
            weight_preference=WeightPreference.HEAVIER,
            height_weight=0.8,
            expertise_weight=0.5,
        )

    assignment = find_optimal_assignment(
        castellers,
        spec,
        all_assignments,
        columns,
        column_tronc_heights,
        optimization_method=optimization_method,
        use_weight=use_weight,
    )

    all_assignments[primeres_key] = assignment
    created[primeres_key] = assignment

    # 2) Subsequent mans rows - skip if primeres_mans is empty
    if not any(assignment.get(col) for col in columns.keys()):
        print(f"  Skipping remaining mans rows - no primeres_mans assigned")
        return created
    
    # Clean implementation: build rows sequentially
    prev_key = primeres_key
    row_idx = 2
    max_rows = min(n_rows, 20)  # Safety limit to prevent infinite loops
    
    while row_idx <= max_rows:
        key = f'mans_row_{row_idx}'

        # Create a spec that references the previous row
        mans_spec = PositionRequirements(
            position_name=key,
            expertise_keywords=['primeres', 'primeres_mans', 'general'],
            count_per_column=1,
            reference_positions=[prev_key],
            height_ratio_min=0.80,  # 80% of previous row
            height_ratio_max=1.00,  # up to equal height
            optimization_objective=OptimizationObjective.HEIGHT_COMPLIANCE,
            weight_preference=WeightPreference.NEUTRAL,
            height_weight=0.9,
            expertise_weight=0.5,
            similarity_weight=0.1,
        )

        assignment = find_optimal_assignment(
            castellers,
            mans_spec,
            all_assignments,
            columns,
            column_tronc_heights,
            optimization_method=optimization_method,
            use_weight=use_weight,
        )

        all_assignments[key] = assignment
        created[key] = assignment
        
        # Check if we still have available candidates for next row
        from .assign import filter_available_castellers
        remaining = filter_available_castellers(castellers, all_assignments)
        
        # Stop only if no more candidates OR we've filled requested rows
        if len(remaining) == 0:
            print(f"  All available castellers assigned after {key}")
            break
        elif row_idx >= n_rows:
            print(f"  Completed requested {n_rows} mans rows. {len(remaining)} castellers still available.")
            break  # Stop after filling requested rows
        
        prev_key = key
        row_idx += 1

    return created


def assign_laterals(
    castellers,
    columns: Dict[str, int],
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]],
    all_assignments: Dict[str, Dict[str, Tuple[str, ...]]],
    lateral_key: str = 'laterals',
    optimization_method: str = 'greedy',
    use_weight: bool = True,
    count_per_column_override: Optional[int] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Assign lateral positions using fixed 51% ratio rule with even distribution.
    
    CRITICAL: Must use ALL remaining available castellers.
    CRITICAL FIX: Enforce even distribution across columns (prevent 3,0,0 pattern).

    Returns assignment dict and updates `all_assignments[lateral_key]`.
    """
    from .assign import filter_available_castellers
    
    # Get available candidates
    available_all = filter_available_castellers(castellers, all_assignments)
    
    if len(available_all) == 0:
        print(f"  No available castellers for {lateral_key}")
        all_assignments[lateral_key] = {col: () for col in columns.keys()}
        return {col: () for col in columns.keys()}
    
    num_columns = len(columns)
    
    # Calculate target distribution for evenness
    total_laterals_needed = count_per_column_override * num_columns if count_per_column_override else len(available_all)
    base_per_column = max(1, total_laterals_needed // num_columns)
    remainder = total_laterals_needed % num_columns
    
    target_distribution = {}
    column_names = list(columns.keys())
    
    # Calculate target distribution: some columns get +1 for remainder
    for i, col_name in enumerate(column_names):
        target_distribution[col_name] = base_per_column + (1 if i < remainder else 0)
    
    print(f"  {lateral_key} target distribution: {target_distribution}")
    
    # Custom balanced assignment algorithm
    assignment = {}
    used_castellers = set()
    
    # Sort columns by target count (highest first for greedy balance)
    sorted_columns = sorted(target_distribution.items(), key=lambda x: -x[1])
    
    for col_name, target_count in sorted_columns:
        if target_count == 0:
            assignment[col_name] = ()
            continue
        
        # Calculate reference height for this column
        from .optimize import _calculate_reference_heights
        temp_spec = PositionRequirements(
            position_name=lateral_key,
            expertise_keywords=['lateral', 'general'],
            count_per_column=1,
            reference_positions=['baix', 'segon'],
            height_ratio_min=0.51,
            height_ratio_max=0.51,
            optimization_objective=OptimizationObjective.HEIGHT_COMPLIANCE,
        )
        
        reference_heights = _calculate_reference_heights(
            columns, temp_spec.reference_positions, all_assignments, column_tronc_heights, castellers
        )
        target_height = reference_heights.get(col_name, 0)
        
        # Score and select best candidates for this column
        column_candidates = available_all[
            ~available_all['Nom complet'].isin(used_castellers)
        ].copy()
        
        if len(column_candidates) < target_count:
            print(f"  ⚠️  WARNING: Only {len(column_candidates)} candidates available for {col_name}, "
                  f"requested {target_count}")
            target_count = len(column_candidates)
        
        # Sort candidates by score
        from .optimize import _calculate_candidate_score
        column_candidates['score'] = column_candidates.apply(
            lambda row: _calculate_candidate_score(row, target_height, temp_spec, [], use_weight),
            axis=1
        )
        
        # Select best candidates
        selected = column_candidates.nsmallest(target_count, 'score')
        selected_names = tuple(selected['Nom complet'].tolist())
        
        assignment[col_name] = selected_names
        used_castellers.update(selected_names)
        
        print(f"    {col_name}: {len(selected_names)} laterals assigned")
    
    all_assignments[lateral_key] = assignment
    return assignment


def assign_daus_or_vents(
    castellers,
    columns: Dict[str, int],
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]],
    all_assignments: Dict[str, Dict[str, Tuple[str, ...]]],
    key: str = 'daus',
    optimization_method: str = 'simulated_annealing',
    use_weight: bool = True,
    count_per_column: Optional[int] = None,  # Auto-calculate to use all available
) -> Dict[str, Tuple[str, ...]]:
    """Assign peripheral "daus/vents" positions that fill gaps between columns.
    
    CRITICAL: Must use ALL available castellers. Auto-calculates count_per_column
    to maximize usage.
 
    Returns the assignment dict and updates `all_assignments[key]`.
    """
    from .assign import filter_available_castellers
    
    MIN_PER_COLUMN = {
        "laterals": 2,
        "daus": 2,
        "mans": 2,
        'agulles': 1,
        'baix': 1,
        'crossa': 2,
        'contrafort': 1,
    }
    MAX_PER_COLUMN = {
        'baix': 1,
        'crossa': 2,
        'contrafort': 1,
        'agulles': 1,
    }
    
    # Create spec FIRST (without count)
    spec = PositionRequirements(
        position_name=key,
        expertise_keywords=['Dau/Vent', 'dau', 'vent', 'dau/vent', 'general'],  # Match actual data format
        count_per_column=1,  # Placeholder, will be recalculated
        reference_positions=['baix', 'segon'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        optimization_objective=OptimizationObjective.HEIGHT_COMPLIANCE,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        weight_factor=0.3,
    )
    
    # Get expertise-filtered AND available candidates
    from .optimize import _get_valid_candidates
    expertise_candidates = _get_valid_candidates(castellers, spec, allow_relaxed=False)
    
    from .assign import filter_available_castellers
    available_all = filter_available_castellers(castellers, all_assignments)
    
    # Intersection: has expertise AND not assigned
    available = expertise_candidates[
        expertise_candidates['Nom complet'].isin(available_all['Nom complet'])
    ]
    
    # NOW calculate count based on actual available candidates
    if count_per_column is None:
        if len(available) == 0:
            print(f"  No candidates with {key} expertise available")
            all_assignments[key] = {col: () for col in columns.keys()}
            return {col: () for col in columns.keys()}
        
        num_columns = len(columns)
        max_possible = len(available) // num_columns
        
        base_per_column = max(MIN_PER_COLUMN.get(key, 1), max_possible)
        remainder = len(available) % num_columns
        
        count_per_column = min(
            base_per_column + (1 if remainder > 0 else 0),
            len(available)  # Cap at total available
        )
        
        print(f"  Auto-calculated {key} count_per_column: {count_per_column} "
              f"({len(available)} available with expertise ÷ {num_columns} columns)")
    
    # Update spec with correct count
    spec = dataclasses.replace(spec, count_per_column=count_per_column)

    assignment = find_optimal_assignment(
        castellers,
        spec,
        all_assignments,
        columns,
        column_tronc_heights,
        optimization_method=optimization_method,
        use_weight=use_weight,
    )

    all_assignments[key] = assignment
    return assignment

def assign_rows_pipeline(
    castellers,
    columns: Dict[str, int],
    column_tronc_heights: Optional[Dict[str, Dict[str, int]]],
    all_assignments: Dict[str, Dict[str, Tuple[str, ...]]],
    mans_rows: int = 3,
    include_laterals: bool = True,
    include_daus: bool = True,
    include_crosses: bool = False,  # Now assigned in main pipeline
    include_contraforts: bool = False,  # Now assigned in main pipeline
    include_agulles: bool = False,  # Now assigned in main pipeline
):
    """High-level convenience wrapper to run the peripheral-row assignment pipeline.

    Correct building order per strategy: baixos, crosses, contraforts, agulles, mans, daus, laterals
    
    Note: crosses, contraforts, and agulles are now assigned in the main pipeline (assign.py)
    This function handles only: mans (all rows), daus, laterals

    Returns a dict with the created assignments for the rows.
    """
    result = {}

    # 1) Sequential mans (primeres_mans + additional rows)
    # CRITICAL: Must maximize usage of all available castellers
    result.update(assign_sequential_mans(
        castellers,
        columns,
        column_tronc_heights,
        all_assignments,
        n_rows=mans_rows,
    ))

    # 2) Daus/Vents - after mans
    if include_daus:
        result['daus'] = assign_daus_or_vents(
            castellers,
            columns,
            column_tronc_heights,
            all_assignments,
        )

    # 3) Laterals - last
    if include_laterals:
        result['laterals'] = assign_laterals(
            castellers,
            columns,
            column_tronc_heights,
            all_assignments,
        )

    # Legacy support - these should not be called anymore as they're in main pipeline
    if include_crosses and 'crossa' in POSITION_SPECS:
        print("  Warning: Crosses should be assigned in main pipeline, not here")
        
    if include_contraforts and 'contrafort' in POSITION_SPECS:
        print("  Warning: Contraforts should be assigned in main pipeline, not here")
        
    if include_agulles and 'agulla' in POSITION_SPECS:
        print("  Warning: Agulles should be assigned in main pipeline, not here")

    return result