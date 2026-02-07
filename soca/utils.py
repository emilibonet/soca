import json
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List

def to_dict(obj, classkey: str = '__class__') -> dict:
    if isinstance(obj, dict):
        return {k: to_dict(v, classkey) for k, v in obj.items()}
    elif hasattr(obj, "_ast"):
        return to_dict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return [to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = {}
        for key, value in obj.__dict__.items():
            if not callable(value) and not key.startswith('_'):
                data[key] = to_dict(value, classkey)
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    return obj


def save(obj, path: str) -> None:
    if not path.endswith('.json'):
        path += '.json'
    with open(path, 'w') as f:
        json.dump(to_dict(obj, '__class__'), f, indent=2)


def from_dict(d, class_map: dict) -> dict:
    if isinstance(d, dict):
        if '__class__' in d:
            cls_name = d.pop('__class__')
            module_name = d.pop('__module__', None)
            if cls_name in class_map:
                cls = class_map[cls_name]
            else:
                if module_name:
                    module = __import__(module_name, fromlist=[cls_name])
                    cls = getattr(module, cls_name)
                else:
                    cls = globals().get(cls_name)
            instance = cls.__new__(cls)
            for key, value in d.items():
                setattr(instance, key, from_dict(value, class_map))
            return instance
        return {k: from_dict(v, class_map) for k, v in d.items()}
    elif isinstance(d, list):
        return [from_dict(v, class_map) for v in d]
    else:
        return d


def load(path: str, class_map: dict):
    if not path.endswith('json'):
        raise ValueError(f"Provided path '{path}' is not a JSON file.")
    with open(path , 'r') as f:
        obj = from_dict(json.load(f), class_map)
    return obj


def build_columns(columns_config) -> Dict[str, float]:
    """
    Convert column layout configuration to base heights for each column.

    Accepts either the legacy dict form (name->num_baixos) or the new
    simplified list form: ["Rengla", "Plena", "Buida"]. Returns a mapping
    of normalized column names to base heights in centimeters.
    """
    COLUMN_BASE_HEIGHTS = {
        1: 175,
        2: 180,
        3: 185,
    }

    # Modifiers keyed by normalized full names
    COLUMN_TYPE_MODIFIERS = {
        "Rengla": 1.00,
        "Plena": 1.00,
        "Buida": 0.95,
    }

    # Map short codes to full names
    COLUMN_NAME_MAP = {
        "R": "Rengla",
        "P": "Plena",
        "B": "Buida",
        "Rengla": "Rengla",
        "Plena": "Plena",
        "Buida": "Buida",
    }

    result: Dict[str, float] = {}

    # Support list input: treat each element as a column name with 1 baix
    if isinstance(columns_config, list):
        iterable = [(c, 1) for c in columns_config]
    elif isinstance(columns_config, dict):
        iterable = list(columns_config.items())
    else:
        raise ValueError("columns_config must be a list or dict of column names")

    for col_name, col_value in iterable:
        normalized_name = COLUMN_NAME_MAP.get(col_name, col_name)

        # Determine number of baixos (default 1)
        num_baixos = 1
        if isinstance(col_value, (int, float)):
            try:
                num_baixos = int(col_value)
            except Exception:
                num_baixos = 1

        base_height = COLUMN_BASE_HEIGHTS.get(num_baixos, 175)

        modifier = COLUMN_TYPE_MODIFIERS.get(normalized_name, 1.0)
        final_height = base_height * modifier

        result[normalized_name] = round(final_height, 1)

    return result


def compute_column_tronc_heights(
    all_assignments: Dict[str, Dict[str, Tuple[Any, ...]]], 
    tronc_positions: list,
    castellers: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate the actual height of tronc positions per column based on assigned castellers.
    
    Parameters
    ----------
    all_assignments : dict
        Mapping of position_name -> {column_name -> tuple(casteller_ids)}
    tronc_positions : list
        Ordered list of tronc position names to include in calculations
    castellers : pd.DataFrame, optional
        DataFrame with casteller data including height information. If None, returns
        placeholder heights.
    
    Returns
    -------
    dict
        Nested mapping {column_name: {position_name: height_cm}} with actual heights
        of assigned castellers.
    """
    result = {}
    
    # Get all column names from the assignments
    column_names = set()
    for position_assignments in all_assignments.values():
        column_names.update(position_assignments.keys())
    
    for column_name in column_names:
        column_heights = {}
        
        for position_name in tronc_positions:
            # Get assignments for this position and column
            position_assignments = all_assignments.get(position_name, {})
            assigned_ids = position_assignments.get(column_name, ())
            
            if not assigned_ids or assigned_ids == ():
                # No assignment - use default height
                column_heights[position_name] = 175.0
                continue
            
            if castellers is None:
                # No casteller data available - use placeholder
                column_heights[position_name] = 175.0
                continue
            
            # Calculate height based on assigned castellers
            total_height = 0.0
            valid_count = 0
            
            for casteller_id in assigned_ids:
                if casteller_id is None:
                    continue
                    
                try:
                    # Try multiple strategies to find the casteller
                    casteller_row = None
                    
                    # Strategy 1: Try by 'id' column if it exists
                    if 'id' in castellers.columns:
                        casteller_row = castellers[castellers['id'] == casteller_id]
                    
                    # Strategy 2: Try by 'Nom complet' (name-based lookup)
                    if (casteller_row is None or len(casteller_row) == 0) and 'Nom complet' in castellers.columns:
                        casteller_row = castellers[castellers['Nom complet'] == casteller_id]
                    
                    # Strategy 3: Try by index
                    if (casteller_row is None or len(casteller_row) == 0):
                        try:
                            casteller_row = castellers.loc[[casteller_id]]
                        except (KeyError, TypeError):
                            pass
                    
                    if casteller_row is not None and len(casteller_row) > 0:
                        # Get height - try different possible column names
                        height = None
                        for height_col in ['Alçada (cm)', 'alçada', 'alcada', 'height', 'altura']:
                            if height_col in casteller_row.columns:
                                height_val = casteller_row[height_col].iloc[0]
                                if pd.notna(height_val) and height_val > 0:
                                    height = float(height_val)
                                    break
                        
                        if height is not None:
                            total_height += height
                            valid_count += 1
                except (KeyError, IndexError, ValueError, TypeError):
                    # Skip if we can't find the casteller or get their height
                    continue
            
            # Use average height if multiple castellers assigned, or total if single
            if valid_count > 0:
                avg_height = total_height / valid_count
                column_heights[position_name] = round(avg_height, 1)
            else:
                # Fallback to default height
                column_heights[position_name] = 175.0
        
        result[column_name] = column_heights
    
    return result

def filter_available_castellers(castellers, all_assignments, base_db_unavailable_flag: Optional[str] = None):
    """Return a filtered DataFrame of castellers excluding those already assigned
    in all_assignments and those flagged as unavailable in the base database.

    CRITICAL FIX: all_assignments contains NAMES (str), not IDs. We must filter by name.
    
    - castellers: pd.DataFrame with casteller data
    - all_assignments: mapping position_name -> {column -> tuple(names)} OR
                       position_name -> {queue_id -> [tuple(names), ...]} for queues
    - base_db_unavailable_flag: optional column name in castellers that if True/1 means not attending
    """
    # Collect assigned NAMES (not IDs)
    assigned_names = set()
    for pos_map in all_assignments.values():
        if not isinstance(pos_map, dict):
            continue
        for col, assignments in pos_map.items():
            if assignments is None:
                continue
            
            # Handle queue structure: list of tuples
            if isinstance(assignments, list):
                for depth_tuple in assignments:
                    if depth_tuple and isinstance(depth_tuple, tuple):
                        for name in depth_tuple:
                            if name is not None:
                                assigned_names.add(str(name) if not isinstance(name, str) else name)
            # Handle standard structure: single tuple
            elif isinstance(assignments, tuple):
                for name in assignments:
                    if name is not None:
                        assigned_names.add(str(name) if not isinstance(name, str) else name)

    # Filter by name (not ID)
    df = castellers[~castellers['Nom complet'].isin(assigned_names)]

    if base_db_unavailable_flag is not None and base_db_unavailable_flag in df.columns:
        df = df[~df[base_db_unavailable_flag].astype(bool)]

    return df


def _get_all_assigned_castellers(previous_assignments: Dict[str, Dict]) -> List[str]:
    """Extract all assigned casteller names from previous assignments."""
    
    assigned = set()
    
    for position_name, assignment in previous_assignments.items():
        for column, castellers in assignment.items():
            if isinstance(castellers, (tuple, list)):
                assigned.update(c for c in castellers if c is not None)
            elif castellers is not None:
                assigned.add(castellers)
    
    return list(assigned)


# Logging helpers
import logging
import os

# ANSI color codes for level names
_LEVEL_COLORS = {
    'CRITICAL': '\u001b[31m',  # red
    'ERROR': '\u001b[31m',     # red
    'WARN': '\u001b[33m',      # yellow (alias)
    'WARNING': '\u001b[33m',   # yellow
    'INFO': '\u001b[34m',      # blue
    'DEBUG': '\u001b[38;5;208m',# orange (256-color)
    'TRACE': '\u001b[90m',     # grey
}
_COLOR_RESET = '\u001b[0m'


class ColoredLevelFormatter(logging.Formatter):
    """Formatter that wraps the entire formatted log line in ANSI color codes."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        text = super().format(record)
        color = _LEVEL_COLORS.get(record.levelname, '')
        if color:
            return f"{color}{text}{_COLOR_RESET}"
        return text


def get_logger(name: str = None) -> logging.Logger:
    """Return a configured logger for the package.

    Uses the `SOCA_LOG_LEVEL` environment variable (defaults to INFO).
    Logs to stderr with a compact formatter.
    """
    lvl_name = os.environ.get('SOCA_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, lvl_name, logging.INFO)

    logger_name = name or 'soca'
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        # already configured
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = '%(message)s'
    formatter = ColoredLevelFormatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
