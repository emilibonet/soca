"""
End-to-end castell assignment pipeline — patched to accept a partial `all_assignments` input.

Behavior changes:
- `build_castell_assignment` now accepts an optional `all_assignments` argument. If
  provided, the pipeline will treat it as pre-filled assignments (for example,
  preassigned tronc members from an Excel layout) and will only fill missing
  columns/positions.
- When assigning tronc positions, the function will call the optimizer but will
  merge results: preassigned entries are preserved and only missing columns are
  populated with newly computed assignments.
- If a `position_name` in `tronc_positions` has no POSITION_SPECS entry and some
  columns are still missing in `all_assignments`, the function raises a
  ValueError so the caller can provide the missing preassignment or add a spec.

This file remains the orchestrator and keeps no heuristics beyond orchestration.
"""
from typing import Dict, Tuple, Optional, Any, List
import unicodedata
import re
import pandas as pd

# Core optimizer
from .optimize import (
    find_optimal_assignment,
    POSITION_SPECS,
)

# Structural helpers (tronc / pinya definitions)
from .castell import (
    build_columns,
    compute_column_tronc_heights,
)

# Peripheral rows (mans, laterals, vents/daus, crosses, contraforts, agulles)
from .optimize_rows import (
    assign_rows_pipeline
)


def build_castell_assignment(
    castellers: pd.DataFrame,
    castell_config: Dict[str, Any],
    optimization_method: str = 'greedy',
    use_weight: bool = True,
    all_assignments: Optional[Dict[str, Dict[str, tuple]]] = None,
) -> Dict[str, Dict]:
    """
    Build a full castell assignment from configuration and casteller data.

    This patched version accepts an optional `all_assignments` dict containing
    preassigned positions (partial or complete). The pipeline will preserve any
    prefilled entries and fill only missing positions/columns.

    Parameters
    ----------
    castellers : pd.DataFrame
        Casteller database
    castell_config : Dict[str,Any]
        Configuration describing the castell. Expected keys:
          - 'columns': column layout spec passed to build_columns
          - 'tronc_positions': ordered list of tronc position names
          - 'mans_rows': int, number of mans rows to generate
          - peripheral flags like 'include_laterals'
          - optional 'base_db_unavailable_flag'
    optimization_method : str
        Optimization backend to use (greedy, exhaustive, simulated_annealing)
    use_weight : bool
        Whether to include weight in the objective where supported
    all_assignments : Optional[Dict[str,Dict[str,tuple]]]
        Optional prefilled assignment map. The function updates this dict in
        place and also returns it.

    Returns
    -------
    Dict[str,Dict]
        all_assignments: mapping position_name -> {column_name -> tuple(casteller_ids)}
    """

    # 1) Build columns and structural metadata
    columns = build_columns(castell_config['columns'])

    # Use or create the container for every assignment produced
    if all_assignments is None:
        all_assignments = {}

    # Ensure the dict has per-position maps present (for merging convenience)
    for pos in castell_config['tronc_positions']:
        all_assignments.setdefault(pos, {})

    # 2) Assign ONLY pinya-level tronc positions (baix only - the base of each column)
    # Other tronc positions (segon, terç, quart, etc.) should be pre-assigned in all_assignments
    # as they form the actual tower structure
    pinya_tronc_positions = ['baix']  # Only baix needs optimization for column balance
    
    for position_name in pinya_tronc_positions:
        if position_name not in castell_config['tronc_positions']:
            continue
        # Skip columns that are already assigned in the provided all_assignments
        already_assigned_columns = set(all_assignments.get(position_name, {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments[position_name].get(c)]

        if not missing_columns:
            # position fully preassigned — nothing to do
            continue

        # If we have a spec for this position, compute assignments for missing columns
        if position_name in POSITION_SPECS:
            spec = POSITION_SPECS[position_name]

            # Request an assignment for the whole set of columns, then merge only missing columns
            computed_assignment = find_optimal_assignment(
                castellers=castellers,
                position_spec=spec,
                previous_assignments=all_assignments,
                columns=columns,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight
            )

            # Merge: preserve existing values, fill missing columns from computed_assignment
            all_assignments.setdefault(position_name, {})
            for col_name, value in computed_assignment.items():
                if col_name not in all_assignments[position_name] or not all_assignments[position_name].get(col_name):
                    all_assignments[position_name][col_name] = value
        else:
            # No spec available: cannot compute missing columns — require caller to prefill
            raise ValueError(
                f"Missing POSITION_SPECS entry for '{position_name}', and the following columns are still unfilled: {missing_columns}. "
                "Provide preassigned values for these columns or add a POSITION_SPECS entry."
            )

    # 2a) Assign crosses immediately after tronc positions (strategic building requirement)
    if 'crossa' in POSITION_SPECS:
        all_assignments.setdefault('crossa', {})
        
        already_assigned_columns = set(all_assignments.get('crossa', {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments['crossa'].get(c)]

        if missing_columns:
            crossa_spec = POSITION_SPECS['crossa']
            
            computed_crossa_assignment = find_optimal_assignment(
                castellers=castellers,
                position_spec=crossa_spec,
                previous_assignments=all_assignments,
                columns=columns,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight
            )

            # Merge: preserve existing values, fill missing columns from computed assignment
            for col_name, value in computed_crossa_assignment.items():
                if col_name not in all_assignments['crossa'] or not all_assignments['crossa'].get(col_name):
                    all_assignments['crossa'][col_name] = value

    # 2b) Assign contrafort after crossa
    if 'contrafort' in POSITION_SPECS:
        all_assignments.setdefault('contrafort', {})
        
        already_assigned_columns = set(all_assignments.get('contrafort', {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments['contrafort'].get(c)]

        if missing_columns:
            contrafort_spec = POSITION_SPECS['contrafort']
            
            computed_contrafort_assignment = find_optimal_assignment(
                castellers=castellers,
                position_spec=contrafort_spec,
                previous_assignments=all_assignments,
                columns=columns,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight
            )

            for col_name, value in computed_contrafort_assignment.items():
                if col_name not in all_assignments['contrafort'] or not all_assignments['contrafort'].get(col_name):
                    all_assignments['contrafort'][col_name] = value

    # 3) Compute per-column tronc heights (used by mans / laterals / agulles)
    # This must happen AFTER we have all tronc positions (either pre-assigned or computed)
    # Pass the castellers DataFrame so we can get actual heights
    column_tronc_heights = compute_column_tronc_heights(
        all_assignments, 
        castell_config['tronc_positions'],
        castellers  # Add castellers parameter
    )
    # 4) Peripheral rows in the specified order — apply filtering before
    if 'agulla' in POSITION_SPECS:
        all_assignments.setdefault('agulla', {})
        
        already_assigned_columns = set(all_assignments.get('agulla', {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments['agulla'].get(c)]

        if missing_columns:
            agulla_spec = POSITION_SPECS['agulla']
            
            computed_agulla_assignment = find_optimal_assignment(
                castellers=castellers,
                position_spec=agulla_spec,
                previous_assignments=all_assignments,
                columns=columns,
                column_tronc_heights=column_tronc_heights,
                optimization_method=optimization_method,
                use_weight=use_weight
            )

            for col_name, value in computed_agulla_assignment.items():
                if col_name not in all_assignments['agulla'] or not all_assignments['agulla'].get(col_name):
                    all_assignments['agulla'][col_name] = value
    
    # 5) Peripheral rows: mans, daus, laterals (in that order per building strategy)
    assign_rows_pipeline(
        castellers=castellers,
        columns=columns,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
        mans_rows=castell_config.get('mans_rows', 2),
        include_laterals=castell_config.get('include_laterals', True),
        include_daus=castell_config.get('include_daus', True),
        include_crosses=False,  # Already assigned above
        include_contraforts=False,  # Already assigned above
        include_agulles=False,  # Already assigned above
    )

    # Validate structure before returning
    errors, warnings = validate_structure(all_assignments, columns)
    
    if errors:
        print("\n" + "="*60)
        print("CRITICAL STRUCTURAL ERRORS:")
        for err in errors:
            print(f"  ❌ {err}")
        print("="*60)
    
    if warnings:
        print("\nWarnings:")
        for warn in warnings:
            print(f"  ⚠️  {warn}")
    
    return all_assignments


def validate_structure(all_assignments: Dict[str, Dict[str, tuple]], columns: Dict[str, int]) -> tuple[List[str], List[str]]:
    """Validate completed castell structure for critical safety issues.
    
    Parameters:
        all_assignments: Complete assignment dict
        columns: Column definitions
        
    Returns:
        tuple: (errors, warnings) lists
    """
    errors = []
    warnings = []
    
    # Check critical positions are filled in ALL columns
    for col_name in columns.keys():
        # Baix (base) - absolutely critical for safety
        if 'baix' in all_assignments:
            baix_assignments = all_assignments['baix'].get(col_name, ())
            baix_count = len([c for c in baix_assignments if c])
            if baix_count == 0:
                errors.append(f"Column {col_name} missing baix (load bearer - CRITICAL)")
            elif baix_count < 1:
                warnings.append(f"Column {col_name} has insufficient baix assignments: {baix_count}")
        
        # Crossa (cross support) - important for stability
        if 'crossa' in all_assignments:
            crossa_assignments = all_assignments['crossa'].get(col_name, ())
            crossa_count = len([c for c in crossa_assignments if c])
            if crossa_count == 0:
                warnings.append(f"Column {col_name} missing crossa (cross support)")
            elif crossa_count < 2:
                warnings.append(f"Column {col_name} has insufficient crossa support: {crossa_count} (recommended: 2)")
        
        # Contrafort (buttress) - lateral stability
        if 'contrafort' in all_assignments:
            contrafort_assignments = all_assignments['contrafort'].get(col_name, ())
            contrafort_count = len([c for c in contrafort_assignments if c])
            if contrafort_count == 0:
                warnings.append(f"Column {col_name} missing contrafort (lateral support)")
        
        # Agulla (needle/pillar) - central support
        if 'agulla' in all_assignments:
            agulla_assignments = all_assignments['agulla'].get(col_name, ())
            agulla_count = len([c for c in agulla_assignments if c])
            if agulla_count == 0:
                errors.append(f"Column {col_name} missing agulla (CRITICAL - castell cannot be safely built)")

        # Primeres mans - critical per manual §7.1
        if 'primeres_mans' in all_assignments:
            pm_assignments = all_assignments['primeres_mans'].get(col_name, ())
            pm_count = len([c for c in pm_assignments if c])
            if pm_count == 0:
                errors.append(f"Column {col_name} missing primeres mans (CRITICAL)")
        
        # Check for empty assignments in positions that should have castellers
        for position_name, assignments in all_assignments.items():
            if position_name in ['baix', 'segon', 'terç', 'quart', 'cinquè']:  # Core tronc
                col_assignments = assignments.get(col_name, ())
                empty_slots = len([c for c in col_assignments if not c])
                if empty_slots > 0:
                    total_slots = len(col_assignments)
                    warnings.append(f"Column {col_name} {position_name}: {empty_slots}/{total_slots} slots empty")
    
    # Check total structure integrity
    from collections import Counter
    all_assigned_ids = []
    for pos_assignments in all_assignments.values():
        for col_assignments in pos_assignments.values():
            if col_assignments:
                all_assigned_ids.extend(c for c in col_assignments if c)
    duplicates = [c for c, count in Counter(all_assigned_ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate assignments (each casteller must appear at most once): {duplicates}")

    # Check total structure integrity
    assigned_columns = set()
    for position_name, assignments in all_assignments.items():
        for col_name, col_assignments in assignments.items():
            if col_assignments and any(c for c in col_assignments):
                assigned_columns.add(col_name)
    
    # Check if any columns are completely empty
    empty_columns = set(columns.keys()) - assigned_columns
    if empty_columns:
        errors.append(f"Completely empty columns: {', '.join(sorted(empty_columns))}")
    
    # Check minimum column count for safety
    if len(assigned_columns) < 2:
        errors.append(f"Insufficient columns for stable structure: {len(assigned_columns)} (minimum: 2)")
    
    return errors, warnings


def _normalize_name(s: str) -> str:
    """Normalize a name for tolerant matching: lower, strip, remove diacritics, collapse spaces and punctuation."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    # remove diacritics
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # remove punctuation except spaces using Unicode-compatible regex
    # Python's re doesn't support \p{P}\p{S}, so we use character classes
    import string
    punctuation_chars = string.punctuation.replace(' ', '')  # keep spaces
    s = ''.join(ch if ch not in punctuation_chars else ' ' for ch in s)
    # fallback punctuation removal using ascii classes for any remaining characters
    s = re.sub(r"[^\w\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_candidates_by_name(df: pd.DataFrame, name: str, name_col: str) -> pd.DataFrame:
    """Return candidate rows matching the name using multiple strategies."""
    # 1) exact
    exact = df[df[name_col] == name]
    if len(exact) > 0:
        return exact

    # 2) case-insensitive exact
    lower = df[df[name_col].str.lower() == name.lower()]
    if len(lower) > 0:
        return lower

    # 3) normalized exact
    norm_name = _normalize_name(name)
    # create normalized column on the fly
    norm_series = df[name_col].astype(str).apply(_normalize_name)
    norm_exact = df[norm_series == norm_name]
    if len(norm_exact) > 0:
        return norm_exact

    # 4) normalized substring match (best-effort)
    contains_mask = norm_series.str.contains(re.escape(norm_name))
    contains = df[contains_mask]
    if len(contains) > 0:
        return contains

    # 5) no matches
    return df.iloc[0:0]


def resolve_name_to_id(
    castellers: pd.DataFrame,
    name: str,
    name_col: str = 'Nom complet',
    id_col: Optional[str] = None,
) -> Any:
    """Resolve a single name to an identifier.

    Returns the value to use as a unique id for an assigned casteller:
    - if `id_col` exists in the DataFrame, the returned value is the id_col value
    - otherwise the returned value is the DataFrame index value

    Raises ValueError with helpful diagnostics on ambiguous or missing matches.
    """
    if name is None:
        raise ValueError("Empty name provided for preassignment")

    if name_col not in castellers.columns:
        raise ValueError(f"Name column '{name_col}' not found in castellers DataFrame")

    candidates = _find_candidates_by_name(castellers, name, name_col)

    if len(candidates) == 1:
        idx = candidates.index[0]
        if id_col is not None and id_col in castellers.columns:
            return castellers.at[idx, id_col]
        return idx

    if len(candidates) > 1:
        # Ambiguous — provide the list of matching names and any available ids
        sample = []
        for rid, row in candidates.head(10).iterrows():
            if id_col is not None and id_col in castellers.columns:
                sample.append((row[name_col], row[id_col]))
            else:
                sample.append((row[name_col], rid))
        raise ValueError(
            f"Ambiguous preassignment for '{name}'. Multiple matches found: {sample}. "
            "Please disambiguate (use exact name as in 'full_name', provide an 'id' column, or adjust the layout)."
        )

    # No matches: offer helpful hint about available names
    example_names = castellers[name_col].astype(str).head(10).tolist()
    raise ValueError(
        f"Preassigned name not found: '{name}'. Examples of names in database: {example_names[:10]}. "
        "Ensure exact spelling or provide an 'id' column in the layout."
    )


def apply_preassigned_to_all_assignments(
    preassigned: Dict[str, Dict[str, Tuple[str, ...]]],
    castellers: pd.DataFrame,
    all_assignments: Dict[str, Dict[str, Tuple[Any, ...]]],
    name_col: str = 'Nom complet',
    id_col: Optional[str] = None,
    assigned_flag_col: str = 'assigned',
) -> None:
    """Apply preassigned layout (by names) to `all_assignments` in-place.

    Behavior:
    - Resolves each name using `resolve_name_to_id`.
    - If `id_col` is provided and exists, the resolved id will be the id_col value;
      otherwise the DataFrame index value is used.
    - Marks resolved castellers as assigned using `assigned_flag_col` on the DataFrame.
    - Populates `all_assignments[pos][column] = tuple(resolved_ids)`.

    Raises:
    - ValueError for ambiguous or missing names with actionable messages.
    """
    # Prepare assigned flag
    if assigned_flag_col not in castellers.columns:
        castellers[assigned_flag_col] = False

    # If id_col not specified but 'id' exists, prefer it for clarity
    if id_col is None and 'id' in castellers.columns:
        id_col = 'id'

    # Ensure all_assignments has containers
    for pos in preassigned.keys():
        all_assignments.setdefault(pos, {})

    for pos, cols in preassigned.items():
        for colname, names_tuple in cols.items():
            if not names_tuple:
                # empty tuple -> leave as empty assignment
                all_assignments[pos][colname] = ()
                continue

            resolved_ids: List[Any] = []
            for name in names_tuple:
                # If the caller accidentally provided numeric ids as strings, try to detect
                # and accept them directly if they match a DataFrame id or index.
                if isinstance(name, (int, float)) and not pd.isna(name):
                    # numeric id provided
                    if id_col is not None and id_col in castellers.columns and int(name) in set(castellers[id_col].astype(int)):
                        resolved_ids.append(int(name))
                        continue
                    elif int(name) in set(castellers.index.astype(int)):
                        resolved_ids.append(int(name))
                        continue
                    # otherwise fallthrough to name resolution

                # Resolve by name (robust)
                resolved = resolve_name_to_id(castellers, str(name), name_col=name_col, id_col=id_col)
                resolved_ids.append(resolved)

            # Set assignment for this column
            all_assignments[pos][colname] = tuple(resolved_ids)

            # Mark assigned in the castellers DataFrame. We need to mark by id_col values
            if id_col is not None and id_col in castellers.columns:
                castellers.loc[castellers[id_col].isin(resolved_ids), assigned_flag_col] = True
            else:
                # resolved_ids are DataFrame indices
                castellers.loc[castellers.index.isin(resolved_ids), assigned_flag_col] = True


# --- Existing helpers below (unchanged) ---

def filter_available_castellers(castellers, all_assignments, base_db_unavailable_flag: Optional[str] = None):
    """Return a filtered DataFrame of castellers excluding those already assigned
    in all_assignments and those flagged as unavailable in the base database.

    CRITICAL FIX: all_assignments contains NAMES (str), not IDs. We must filter by name.
    
    - castellers: pd.DataFrame with casteller data
    - all_assignments: mapping position_name -> {column -> tuple(names)}
    - base_db_unavailable_flag: optional column name in castellers that if True/1 means not attending
    """
    # Collect assigned NAMES (not IDs)
    assigned_names = set()
    for pos_map in all_assignments.values():
        for col, tpl in pos_map.items():
            if tpl is None:
                continue
            for name in tpl:
                if name is None:
                    continue
                # Handle mixed content: some entries might be IDs, others names
                if isinstance(name, str):
                    assigned_names.add(name)
                else:
                    # Convert non-string to string for safety
                    assigned_names.add(str(name))

    # Filter by name (not ID)
    df = castellers[~castellers['Nom complet'].isin(assigned_names)]

    if base_db_unavailable_flag is not None and base_db_unavailable_flag in df.columns:
        df = df[~df[base_db_unavailable_flag].astype(bool)]

    return df

