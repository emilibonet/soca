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
from .utils import get_logger

logger = get_logger(__name__)

from .optimize import find_optimal_assignment

from .utils import (
    build_columns,
    compute_column_tronc_heights,
)

from .queue_assign import assign_rows_pipeline

from .data import (
    POSITION_SPECS
)

# Update build_castell_assignment() - replace column references

def build_castell_assignment(
    castellers: pd.DataFrame,
    castell_config: Dict[str, Any],
    optimization_method: str = 'greedy',
    use_weight: bool = True,
    all_assignments: Optional[Dict[str, Dict[str, tuple]]] = None,
) -> Dict[str, Dict]:
    """Build castell assignment with new queue-based structure."""
    
    # Build columns with normalized names (Rengla/Plena/Buida)
    columns = build_columns(castell_config['columns'])
    
    if all_assignments is None:
        all_assignments = {}
    
    for pos in castell_config['tronc_positions']:
        all_assignments.setdefault(pos, {})
    
    # Assign pinya-level tronc (baix only)
    pinya_tronc_positions = ['baix']
    
    for position_name in pinya_tronc_positions:
        if position_name not in castell_config['tronc_positions']:
            continue
        
        already_assigned_columns = set(all_assignments.get(position_name, {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments[position_name].get(c)]
        
        if not missing_columns:
            continue
        
        if position_name in POSITION_SPECS:
            spec = POSITION_SPECS[position_name]
            computed_assignment = find_optimal_assignment(
                castellers=castellers,
                position_spec=spec,
                previous_assignments=all_assignments,
                columns=columns,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight
            )
            
            all_assignments.setdefault(position_name, {})
            for col_name, value in computed_assignment.items():
                if col_name not in all_assignments[position_name] or not all_assignments[position_name].get(col_name):
                    all_assignments[position_name][col_name] = value
        else:
            raise ValueError(
                f"Missing POSITION_SPECS entry for '{position_name}', unfilled columns: {missing_columns}"
            )
    
    # Assign crosses
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
            
            for col_name, value in computed_crossa_assignment.items():
                if col_name not in all_assignments['crossa'] or not all_assignments['crossa'].get(col_name):
                    all_assignments['crossa'][col_name] = value
    
    # Assign contrafort
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
    
    # Compute tronc heights
    column_tronc_heights = compute_column_tronc_heights(
        all_assignments,
        castell_config['tronc_positions'],
        castellers
    )
    
    # Assign agulles
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
    
    
    peripheral_stats = assign_rows_pipeline(
        castellers=castellers,
        columns=columns,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
        mans_rows=castell_config.get('mans_rows', 2),
        include_laterals=castell_config.get('include_laterals', True),
        include_daus=castell_config.get('include_daus', True)
    )
    
    # Print summaries
    _print_queue_summary(all_assignments, castellers)
    
    # Validate
    errors, warnings = validate_structure(all_assignments, columns)
    
    if errors:
        logger.error("="*60)
        logger.error("CRITICAL STRUCTURAL ERRORS:")
        for err in errors:
            logger.error("✗ %s", err)
        logger.error("="*60)
    
    if warnings:
        logger.warning("Warnings:")
        for warn in warnings:
            logger.warning("⚠️ %s", warn)
    
    return all_assignments


def _print_queue_summary(all_assignments: Dict, castellers: pd.DataFrame):
    """Print formatted summary of queue assignments."""
    
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type not in all_assignments:
            continue
        
        print(f"\n##### {queue_type.upper()} #####")
        
        for queue_id, depth_list in all_assignments[queue_type].items():
            print(f"\n{queue_id}:")
            
            for depth_idx, assignment in enumerate(depth_list, start=1):
                if assignment and assignment[0]:
                    name = assignment[0]
                    h = castellers[castellers['Nom complet'] == name]['Alçada (cm)'].iloc[0]
                    print(f"  Depth {depth_idx}: {name:<16} {h:3.0f} cm")
                else:
                    print(f"  Depth {depth_idx}: [empty]")


def validate_structure(all_assignments: Dict[str, Dict], columns: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """Validate completed castell structure."""
    errors = []
    warnings = []
    
    # Check critical positions
    for col_name in columns.keys():
        # Baix
        if 'baix' in all_assignments:
            baix_assignments = all_assignments['baix'].get(col_name, ())
            baix_count = len([c for c in baix_assignments if c])
            if baix_count == 0:
                errors.append(f"Column {col_name} missing baix (CRITICAL)")
        
        # Crossa
        if 'crossa' in all_assignments:
            crossa_assignments = all_assignments['crossa'].get(col_name, ())
            crossa_count = len([c for c in crossa_assignments if c])
            if crossa_count < 2:
                warnings.append(f"Column {col_name} has {crossa_count} crossa (recommended: 2)")
        
        # Contrafort
        if 'contrafort' in all_assignments:
            contrafort_assignments = all_assignments['contrafort'].get(col_name, ())
            contrafort_count = len([c for c in contrafort_assignments if c])
            if contrafort_count == 0:
                warnings.append(f"Column {col_name} missing contrafort")
        
        # Agulla
        if 'agulla' in all_assignments:
            agulla_assignments = all_assignments['agulla'].get(col_name, ())
            agulla_count = len([c for c in agulla_assignments if c])
            if agulla_count == 0:
                errors.append(f"Column {col_name} missing agulla (CRITICAL)")
    
    # Check queue balance
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type not in all_assignments:
            continue
        
        depths = []
        for queue_id, depth_list in all_assignments[queue_type].items():
            filled_depth = len([d for d in depth_list if d and d[0]])
            depths.append(filled_depth)
        
        if depths:
            variance = max(depths) - min(depths)
            if variance > 1:
                warnings.append(f"{queue_type} queue imbalance: variance={variance} (max=1)")
    
    # Check duplicates
    from collections import Counter
    all_assigned_ids = []
    
    for pos_name, pos_data in all_assignments.items():
        if pos_name in ['mans', 'daus', 'laterals']:
            # Queue structure
            for queue_id, depth_list in pos_data.items():
                for assignment in depth_list:
                    if assignment and assignment[0]:
                        all_assigned_ids.append(assignment[0])
        else:
            # Standard structure
            for col_assignments in pos_data.values():
                if col_assignments:
                    all_assigned_ids.extend(c for c in col_assignments if c)
    
    duplicates = [c for c, count in Counter(all_assigned_ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate assignments: {duplicates}")
    
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

            # Map resolved ids/indices back to the canonical name (name_col) for storage
            resolved_names: List[str] = []
            for rid in resolved_ids:
                name_to_store = None
                # If resolver returned a string, assume it's already a name
                if isinstance(rid, str):
                    name_to_store = rid
                else:
                    # Try id_col lookup first (preferred)
                    if id_col is not None and id_col in castellers.columns:
                        matches = castellers[castellers[id_col] == rid]
                        if not matches.empty:
                            name_to_store = matches[name_col].iloc[0]
                    # Fallback: treat rid as DataFrame index
                    if name_to_store is None:
                        try:
                            matches = castellers.loc[[rid]]
                            if not matches.empty:
                                name_to_store = matches[name_col].iloc[0]
                        except Exception:
                            name_to_store = None

                # As a last resort, stringize the resolved value
                if name_to_store is None:
                    name_to_store = str(rid)

                resolved_names.append(name_to_store)

            # Store canonical names in all_assignments so downstream filters can
            # always compare against the `Nom complet` column.
            all_assignments[pos][colname] = tuple(resolved_names)

            # Mark assigned in the castellers DataFrame. Prefer marking by id
            # where available, but also mark by name for any string-resolved values.
            if id_col is not None and id_col in castellers.columns:
                id_values = [r for r in resolved_ids if not isinstance(r, str)]
                if id_values:
                    castellers.loc[castellers[id_col].isin(id_values), assigned_flag_col] = True
                name_values = [r for r in resolved_ids if isinstance(r, str)]
                if name_values:
                    castellers.loc[castellers[name_col].isin(name_values), assigned_flag_col] = True
            else:
                # No id_col: resolved_ids should be DataFrame indices or names
                idx_values = [r for r in resolved_ids if not isinstance(r, str)]
                if idx_values:
                    castellers.loc[castellers.index.isin(idx_values), assigned_flag_col] = True
                name_values = [r for r in resolved_ids if isinstance(r, str)]
                if name_values:
                    castellers.loc[castellers[name_col].isin(name_values), assigned_flag_col] = True
