"""
End-to-end castell assignment pipeline — patched to accept a partial `all_assignments` input.

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
from .data import POSITION_SPECS
from .display import summarize_assignments
from .queue_assign import assign_rows_pipeline
from .state import AssignmentState, DuplicateAssignmentError

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
    # Collect optimization stats per position (final score, iterations, etc.)
    assignment_stats: Dict[str, Dict] = {}
    
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
            columns_subset = {c: columns[c] for c in missing_columns}
            computed_assignment, stats = find_optimal_assignment(
                castellers=castellers,
                position_spec=spec,
                previous_assignments=all_assignments,
                columns=columns_subset,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight,
                return_stats=True
            )
            assignment_stats[position_name] = stats
            
            all_assignments.setdefault(position_name, {})
            for col_name, value in computed_assignment.items():
                if col_name not in all_assignments[position_name] or not all_assignments[position_name].get(col_name):
                    all_assignments[position_name][col_name] = value
        else:
            raise ValueError(
                f"Missing POSITION_SPECS entry for '{position_name}', unfilled columns: {missing_columns}"
            )
    
    # Assign crosses
    include_crossa = castell_config.get('include_crossa', True)
    if include_crossa and 'crossa' in POSITION_SPECS:
        all_assignments.setdefault('crossa', {})
        already_assigned_columns = set(all_assignments.get('crossa', {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments['crossa'].get(c)]
        
        if missing_columns:
            crossa_spec = POSITION_SPECS['crossa']
            columns_subset = {c: columns[c] for c in missing_columns}
            computed_crossa_assignment, stats = find_optimal_assignment(
                castellers=castellers,
                position_spec=crossa_spec,
                previous_assignments=all_assignments,
                columns=columns_subset,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight,
                return_stats=True
            )
            assignment_stats['crossa'] = stats
            
            for col_name, value in computed_crossa_assignment.items():
                if col_name not in all_assignments['crossa'] or not all_assignments['crossa'].get(col_name):
                    all_assignments['crossa'][col_name] = value
    
    # Assign contrafort
    include_contraforts = castell_config.get('include_contraforts', True)
    if include_contraforts and 'contrafort' in POSITION_SPECS:
        all_assignments.setdefault('contrafort', {})
        already_assigned_columns = set(all_assignments.get('contrafort', {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments['contrafort'].get(c)]
        
        if missing_columns:
            contrafort_spec = POSITION_SPECS['contrafort']
            columns_subset = {c: columns[c] for c in missing_columns}
            computed_contrafort_assignment, stats = find_optimal_assignment(
                castellers=castellers,
                position_spec=contrafort_spec,
                previous_assignments=all_assignments,
                columns=columns_subset,
                column_tronc_heights=None,
                optimization_method=optimization_method,
                use_weight=use_weight,
                return_stats=True
            )
            assignment_stats['contrafort'] = stats
            
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
    include_agulles = castell_config.get('include_agulles', True)
    if include_agulles and 'agulla' in POSITION_SPECS:
        all_assignments.setdefault('agulla', {})
        already_assigned_columns = set(all_assignments.get('agulla', {}).keys())
        missing_columns = [c for c in columns.keys() if c not in already_assigned_columns or not all_assignments['agulla'].get(c)]
        
        if missing_columns:
            agulla_spec = POSITION_SPECS['agulla']
            columns_subset = {c: columns[c] for c in missing_columns}
            computed_agulla_assignment, stats = find_optimal_assignment(
                castellers=castellers,
                position_spec=agulla_spec,
                previous_assignments=all_assignments,
                columns=columns_subset,
                column_tronc_heights=column_tronc_heights,
                optimization_method=optimization_method,
                use_weight=use_weight,
                return_stats=True
            )
            assignment_stats['agulla'] = stats
            
            for col_name, value in computed_agulla_assignment.items():
                if col_name not in all_assignments['agulla'] or not all_assignments['agulla'].get(col_name):
                    all_assignments['agulla'][col_name] = value
    
    
    # Collect peripheral assignment results and stats
    peripheral_result, peripheral_stats = assign_rows_pipeline(
        castellers=castellers,
        columns=columns,
        column_tronc_heights=column_tronc_heights,
        all_assignments=all_assignments,
        mans=castell_config.get('mans', 3),
        daus=castell_config.get('daus', 3),
        laterals=castell_config.get('laterals', 5),
        include_laterals=castell_config.get('include_laterals', True),
        include_daus=castell_config.get('include_daus', True),
        include_mans=castell_config.get('include_mans', True)
    )
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type in peripheral_result:
            all_assignments[queue_type] = peripheral_result[queue_type]
    
    logger.info("\n" + "="*60)
    logger.info("# FINAL ASSIGNMENTS")
    logger.info("="*60)

    summary = summarize_assignments(
        all_assignments=all_assignments,
        castellers=castellers,
        columns=columns,
        column_tronc_heights=column_tronc_heights,
        assignment_stats=assignment_stats,
        peripheral_stats=peripheral_stats
    )
    
    print(summary)
    return all_assignments


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
    raise ValueError(f"Name not found as available casteller.")


def apply_preassigned_to_all_assignments(
    preassigned: Dict[str, Dict[str, Tuple[str, ...]]],
    castellers: pd.DataFrame,
    all_assignments,
    name_col: str = 'Nom complet',
    id_col: Optional[str] = None,
    assigned_flag_col: str = 'assignat',
    availability_flag_col: str = 'Assaig',
    logger_override=None,
    state: Optional[AssignmentState] = None,
    on_conflict: str = 'raise',
    swapped_names: Optional[set] = None,
    failed_preassignments: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None,
) -> None:
    """Apply preassigned layout (by names) to assignments.

    Accepts either a plain dict ``all_assignments`` (legacy) or an
    ``AssignmentState`` instance via *state*.  When *state* is provided,
    duplicate detection is automatic and fail-fast.

    Parameters
    ----------
    on_conflict : str
        ``'raise'`` (default) — raise ``DuplicateAssignmentError`` on
        duplicate name.  ``'warn'`` — log a warning and skip the
        conflicting name (used for peripheral phase 2 where tronc
        assignments take priority).

    Resilience rules (per user requirements):
    - **Duplicates across positions**: raise ``DuplicateAssignmentError``
      (fail fast) — unless *on_conflict='warn'* (peripheral phase).
    - **Name not found in castellers**: warn, skip that name and leave
      the slot empty so the optimiser can fill it later.
    - **Casteller not available** (availability flag False): warn, skip.

    Populates ``state`` (or ``all_assignments``) with resolved **names**
    (not IDs).
    """
    from .data import POSITION_SPECS
    _log = logger_override if logger_override is not None else logger

    # Normalise inputs --------------------------------------------------
    if state is None:
        # Legacy path: wrap dict in an AssignmentState for uniform handling
        state = AssignmentState.from_dict(all_assignments)
        _legacy_dict = all_assignments  # keep ref so we can sync back
    else:
        _legacy_dict = None

    if assigned_flag_col not in castellers.columns:
        castellers[assigned_flag_col] = False

    if id_col is None and 'id' in castellers.columns:
        id_col = 'id'

    # ----------------------------------------------------------------
    # Resolve names → canonical names
    # ----------------------------------------------------------------
    for pos, cols in preassigned.items():
        state.ensure_position(pos)
        position_spec = POSITION_SPECS.get(pos)

        for colname, names_tuple in cols.items():
            if not names_tuple:
                continue

            resolved_names: List[Optional[str]] = []
            for slot_idx, name in enumerate(names_tuple):
                # --- placeholder slot (None / "") → preserve position ---
                if name is None or (isinstance(name, str) and not name.strip()):
                    resolved_names.append(None)
                    continue

                # --- availability check --------------------------------
                if availability_flag_col in castellers.columns:
                    name_matches = castellers[castellers[name_col] == str(name)]
                    if not name_matches.empty:
                        is_unavailable = name_matches[availability_flag_col].iloc[0]
                        if pd.notna(is_unavailable) and not bool(is_unavailable):
                            _log.warning(
                                "PREASSIGNED '%s' for %s[%s] is NOT AVAILABLE "
                                "(%s=False). Removing preassignment.",
                                name, pos, colname, availability_flag_col,
                            )
                            if failed_preassignments is not None:
                                failed_preassignments.setdefault(pos, {}).setdefault(colname, {})[slot_idx] = str(name)
                            resolved_names.append(None)
                            continue

                # --- resolve name --------------------------------------
                try:
                    resolved = resolve_name_to_id(
                        castellers, str(name), name_col=name_col, id_col=id_col
                    )
                except ValueError as e:
                    # Resilience: name not in DB → skip, allow optimiser fill
                    _log.warning(
                        "Skipping preassignment '%s' for %s[%s]: %s",
                        name, pos, colname, e,
                    )
                    if failed_preassignments is not None:
                        failed_preassignments.setdefault(pos, {}).setdefault(colname, {})[slot_idx] = str(name)
                    resolved_names.append(None)
                    continue

                # --- map back to canonical name -----------------------
                canonical = _resolve_to_canonical_name(
                    resolved, castellers, name_col, id_col
                )

                # --- expertise validation (warning only) ---------------
                if position_spec is not None:
                    _warn_if_lacking_expertise(
                        canonical, pos, colname, position_spec, castellers,
                        name_col, _log,
                    )

                resolved_names.append(canonical)

            if not any(n is not None for n in resolved_names):
                continue

            # --- conflict pre-filter (warn mode) -----------------------
            if on_conflict == 'warn':
                filtered = []
                for rn_idx, rn in enumerate(resolved_names):
                    if rn is None:
                        filtered.append(None)
                    elif state.is_assigned(rn):
                        previous = state.find_assignment(rn)
                        _log.warning(
                            "Skipping preassignment '%s' for %s[%s]: "
                            "already assigned to %s (tronc takes priority).",
                            rn, pos, colname, previous,
                        )
                        if swapped_names is not None:
                            swapped_names.add(rn)
                        if failed_preassignments is not None:
                            failed_preassignments.setdefault(pos, {}).setdefault(colname, {})[rn_idx] = rn
                        filtered.append(None)
                    else:
                        filtered.append(rn)
                resolved_names = filtered
                if not any(n is not None for n in resolved_names):
                    continue

            # --- store in AssignmentState (fail-fast on duplicates) -----
            try:
                if pos in ('mans', 'daus', 'laterals'):
                    # Sort real names by height descending (tallest →
                    # depth 1); None placeholders stay in position.
                    real_names = [n for n in resolved_names if n is not None]
                    sorted_real = _sort_names_by_height(
                        real_names, castellers, name_col
                    )
                    # Rebuild list preserving None positions
                    it = iter(sorted_real)
                    final = [next(it) if n is not None else None
                             for n in resolved_names]
                    depth_list = [
                        (n,) if n else (None,) for n in final
                    ]
                    state.assign_queue(pos, colname, depth_list)
                else:
                    state.assign_tronc(pos, colname, tuple(resolved_names))
            except DuplicateAssignmentError as e:
                if on_conflict == 'warn':
                    _log.warning("Conflict during peripheral store: %s", e)
                else:
                    raise DuplicateAssignmentError(
                        f"Preassignment conflict: {e}"
                    ) from None

            # --- mark in DataFrame ------------------------------------
            real_assigned = [n for n in resolved_names if n is not None]
            if real_assigned:
                castellers.loc[
                    castellers[name_col].isin(real_assigned),
                    assigned_flag_col,
                ] = True

    # Sync back to legacy dict if caller used the old API
    if _legacy_dict is not None:
        _legacy_dict.clear()
        _legacy_dict.update(state.to_dict())


# ── helpers for apply_preassigned ─────────────────────────────────────

def _resolve_to_canonical_name(
    resolved, castellers: pd.DataFrame, name_col: str, id_col: Optional[str]
) -> str:
    """Convert a resolved id/index/string back to the canonical name."""
    if isinstance(resolved, str):
        return resolved

    # Try id_col lookup
    if id_col is not None and id_col in castellers.columns:
        matches = castellers[castellers[id_col] == resolved]
        if not matches.empty:
            return matches[name_col].iloc[0]

    # Fallback: DataFrame index
    try:
        matches = castellers.loc[[resolved]]
        if not matches.empty:
            return matches[name_col].iloc[0]
    except Exception:
        pass

    return str(resolved)


def _warn_if_lacking_expertise(
    name: str,
    pos: str,
    colname: str,
    spec,
    castellers: pd.DataFrame,
    name_col: str,
    _log,
) -> None:
    """Emit a warning if the casteller lacks required expertise for *pos*."""
    row = castellers[castellers[name_col] == name]
    if row.empty:
        return
    pos1 = str(row['Posició 1'].iloc[0]).lower() if 'Posició 1' in row.columns else ''
    pos2 = str(row['Posició 2'].iloc[0]).lower() if 'Posició 2' in row.columns else ''
    has_exp = any(
        kw.lower() in pos1 or kw.lower() in pos2
        for kw in spec.expertise_keywords
    )
    if not has_exp:
        _log.warning(
            "PREASSIGNED %s '%s' in column %s lacks required expertise "
            "(keywords: %s)  |  Posició 1='%s', Posició 2='%s'",
            pos.upper(), name, colname, spec.expertise_keywords,
            row['Posició 1'].iloc[0] if 'Posició 1' in row.columns else 'N/A',
            row['Posició 2'].iloc[0] if 'Posició 2' in row.columns else 'N/A',
        )


def _sort_names_by_height(
    names: List[str],
    castellers: pd.DataFrame,
    name_col: str = 'Nom complet',
) -> List[str]:
    """Sort names by height descending (tallest first for queue depth 1).

    Names without a height record in the DataFrame are placed at the end.
    """
    def _height(name: str) -> float:
        row = castellers[castellers[name_col] == name]
        if row.empty or 'Alçada (cm)' not in row.columns:
            return 0.0
        val = row['Alçada (cm)'].iloc[0]
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0

    return sorted(names, key=_height, reverse=True)