import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*include_tailing_empty.*",
    category=UserWarning
)   # Already dealing with empty headers.

import os
import pygsheets
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv



class WeightPreference(Enum):
    HEAVIER = "heavier"
    LIGHTER = "lighter"
    NEUTRAL = "neutral"

class OptimizationObjective(Enum):
    COLUMN_BALANCE = "column_balance"          # Minimize variance across columns (baix)
    EVEN_DISTRIBUTION = "even_distribution"    # Minimize variance across columns (crosses, contraforts, laterals)
    FILL_ALL_REQUIRED = "fill_all_required"    # Every slot filled before optimizing quality (agulles)
    HEIGHT_COMPLIANCE = "height_compliance"    # Match height ratios (primeres mans, mans rows)

@dataclass
class PositionRequirements:
    """
    Unified position requirements specification.
    """
    position_name: str
    expertise_keywords: List[str]  # ["baix"] or ["crossa", "cross"]
    count_per_column: int
    
    # Height calculation
    reference_positions: List[str]  # ["baix"] or ["baix", "segon"]
    height_ratio_min: float
    height_ratio_max: float
    
    # Optimization
    optimization_objective: OptimizationObjective
    weight_preference: WeightPreference = WeightPreference.NEUTRAL
    
    # Scoring weights
    height_weight: float = 1.0
    expertise_weight: float = 0.5
    height_similarity_weight: float = 0.3
    weight_factor: float = 0.2
    height_penalty_factor: float = 0.1
    
    # Optional constraints
    min_experience_level: int = 0  # 0=any, 1=secondary, 2=primary only

# Add new QueueSpec dataclass after WeightPreference
@dataclass
class QueueSpec:
    """Specification for a queue position (mans, daus, laterals)."""
    queue_id: str
    queue_type: str
    column_refs: List[str]
    expertise_keywords: List[str]
    height_ratio_min: float
    height_ratio_max: float
    queue_height_ratio_min: float = 0.80  # For depth > 1
    queue_height_ratio_max: float = 1.00  # For depth > 1
    weight_preference: WeightPreference = WeightPreference.NEUTRAL
    height_weight: float = 0.8
    expertise_weight: float = 0.5
    height_similarity_weight: float = 0.1
    weight_factor: float = 0.2
    # Maximum allowed depth for this queue (can be overridden via YAML)
    max_depth: int = 3
    # Penalty factor for queue height deviations (applies when converted to PositionRequirements)
    height_penalty_factor: float = 0.1

# Queue specifications for standard 3-column castell
MANS_QUEUE_SPECS = {
    'Rengla': QueueSpec(
        queue_id='Rengla',
        queue_type='mans',
        column_refs=['Rengla'],
        expertise_keywords=['primeres'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Plena': QueueSpec(
        queue_id='Plena',
        queue_type='mans',
        column_refs=['Plena'],
        expertise_keywords=['primeres'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Buida': QueueSpec(
        queue_id='Buida',
        queue_type='mans',
        column_refs=['Buida'],
        expertise_keywords=['primeres'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
}

DAUS_QUEUE_SPECS = {
    'R↔P': QueueSpec(
        queue_id='R↔P',
        queue_type='daus',
        column_refs=['Rengla', 'Plena'],
        expertise_keywords=['Dau/Vent'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.3
    ),
    'P↔B': QueueSpec(
        queue_id='P↔B',
        queue_type='daus',
        column_refs=['Plena', 'Buida'],
        expertise_keywords=['Dau/Vent'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.3
    ),
    'B↔R': QueueSpec(
        queue_id='B↔R',
        queue_type='daus',
        column_refs=['Buida', 'Rengla'],
        expertise_keywords=['Dau/Vent'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.3
    ),
}

LATERALS_QUEUE_SPECS = {
    'Rengla-left': QueueSpec(
        queue_id='Rengla-left',
        queue_type='laterals',
        column_refs=['Rengla'],
        expertise_keywords=['lateral'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Rengla-right': QueueSpec(
        queue_id='Rengla-right',
        queue_type='laterals',
        column_refs=['Rengla'],
        expertise_keywords=['lateral'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Plena-left': QueueSpec(
        queue_id='Plena-left',
        queue_type='laterals',
        column_refs=['Plena'],
        expertise_keywords=['lateral'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Plena-right': QueueSpec(
        queue_id='Plena-right',
        queue_type='laterals',
        column_refs=['Plena'],
        expertise_keywords=['lateral'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Buida-left': QueueSpec(
        queue_id='Buida-left',
        queue_type='laterals',
        column_refs=['Buida'],
        expertise_keywords=['lateral'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
    'Buida-right': QueueSpec(
        queue_id='Buida-right',
        queue_type='laterals',
        column_refs=['Buida'],
        expertise_keywords=['lateral'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        queue_height_ratio_min=0.80,
        queue_height_ratio_max=1.00,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.1,
        height_penalty_factor=0.1,
        weight_factor=0.2
    ),
}

# Update POSITION_SPECS to use new column names
POSITION_SPECS = {
    'baix': PositionRequirements(
        position_name='baix',
        expertise_keywords=['baix'],
        count_per_column=1,
        reference_positions=[],
        height_ratio_min=1.0,
        height_ratio_max=1.0,
        optimization_objective=OptimizationObjective.COLUMN_BALANCE,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.0,
        expertise_weight=1.0,
        weight_factor=0.5,
        height_penalty_factor=0.1
    ),

    'crossa': PositionRequirements(
        position_name='crossa',
        expertise_keywords=['crossa'],
        count_per_column=2,
        reference_positions=['baix'],
        height_ratio_min=0.90,
        height_ratio_max=0.95,
        optimization_objective=OptimizationObjective.EVEN_DISTRIBUTION,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.5,
        weight_factor=0.2,
        height_penalty_factor=0.1
    ),

    'contrafort': PositionRequirements(
        position_name='contrafort',
        expertise_keywords=['contrafort'],
        count_per_column=1,
        reference_positions=['baix'],
        height_ratio_min=1.00,
        height_ratio_max=1.05,
        optimization_objective=OptimizationObjective.EVEN_DISTRIBUTION,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=1.0,
        expertise_weight=0.5,
        height_similarity_weight=0.3,
        weight_factor=0.3,
        height_penalty_factor=0.1
    ),

    'agulla': PositionRequirements(
        position_name='agulla',
        expertise_keywords=['agulla'],
        count_per_column=1,
        reference_positions=['baix', 'segon'],
        height_ratio_min=0.50,
        height_ratio_max=0.515,
        optimization_objective=OptimizationObjective.FILL_ALL_REQUIRED,
        weight_preference=WeightPreference.LIGHTER,
        height_weight=1.0,
        expertise_weight=0.5,
        weight_factor=0.3,
        height_penalty_factor=0.1
    ),
}

@dataclass
class Casteller:
    """Canonical Casteller model. Internal attribute names are Python-friendly;
    `to_record()` maps them to the DB column names used by the rest of the codebase.
    """
    nom_complet: str
    nom: Optional[str] = None
    cognoms: Optional[str] = None
    alcada_cm: Optional[float] = None
    alcada_espatlles_cm: Optional[float] = None
    posicio_1: Optional[str] = None
    posicio_2: Optional[str] = None
    assignat: Optional[bool] = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Casteller':
        # Accept either canonical column names or pythonic keys
        def get(k, alt=None):
            return d.get(k, d.get(alt)) if alt is not None else d.get(k)

        return Casteller(
            nom_complet=get('Nom complet', 'nom_complet') or get('Nom', 'nom'),
            nom=get('Nom', 'nom'),
            cognoms=get('Cognoms', 'cognoms'),
            alcada_cm= get('Alçada (cm)', 'alcada_cm'),
            alcada_espatlles_cm= get('Alçada espatlles (cm)', 'alcada_espatlles_cm'),
            posicio_1=get('Posició 1', 'posicio_1'),
            posicio_2=get('Posició 2', 'posicio_2'),
            assignat=get('assignat', 'assignat') in [True, 'True', 'true', 1]
        )

    def to_record(self, name_col: str = 'Nom complet') -> Dict[str, Any]:
        return {
            name_col: self.nom_complet,
            'Nom': self.nom or '',
            'Cognoms': self.cognoms or '',
            'Alçada (cm)': self.alcada_cm,
            'Alçada espatlles (cm)': self.alcada_espatlles_cm,
            'Posició 1': self.posicio_1 or '',
            'Posició 2': self.posicio_2 or '',
            'assignat': self.assignat if self.assignat is not None else ''
        }

def get_castellers() -> pd.DataFrame:
    load_dotenv()
    df = pygsheets.authorize(
        service_file=os.getenv("SERVICE_ACCOUNT_AUTH")
    ).open_by_key(
        os.getenv("FILE_KEY")
    ).worksheet(
        "title", "Base de dades"
    ).get_as_df()
    # Retallar taula
    cut_col = "Funció addicional"
    if cut_col in df.columns:
        cut_idx = df.columns.get_loc(cut_col)
        df = df.iloc[:, :cut_idx]
    else:
        # Fallback: retallar a partir de la primera columna buida
        empty_cols = df.columns[df.isna().all()]
        if len(empty_cols) > 0:
            first_empty_col = empty_cols[0]
            cut_idx = df.columns.get_loc(first_empty_col)
            df = df.iloc[:, :cut_idx]
    df["assignat"] = False
    return df[df["Assaig"] == "SI"].drop(
        columns=[
            "Inactius", "Assaig", "Sobrenom", "Posicio 3"
        ], errors="ignore"
    )


def add_castellers(df: pd.DataFrame, new_castellers: List[Casteller], name_col: str = 'Nom complet') -> pd.DataFrame:
    """Append a list of Casteller instances to `df`, validating names and
    preserving numeric/bool dtypes from the original frame where possible.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    existing_names = set(df.get(name_col, pd.Series(dtype=object)).astype(str).fillna('').tolist())

    processed_rows: List[Dict[str, Any]] = []
    seen_new_names = set()

    for i, item in enumerate(new_castellers):
        if not isinstance(item, Casteller):
            raise TypeError(f'Item at index {i} is not a Casteller instance')

        name_val = item.nom_complet
        if not name_val or str(name_val).strip() == '':
            raise ValueError(f"Casteller at index {i} missing required name")
        if name_val in existing_names:
            raise ValueError(f"Duplicate 'Nom complet' found: '{name_val}' already in DataFrame")
        if name_val in seen_new_names:
            raise ValueError(f"Duplicate 'Nom complet' in new_castellers: '{name_val}'")

        seen_new_names.add(name_val)
        processed_rows.append(item.to_record(name_col=name_col))

    new_df = pd.DataFrame(processed_rows)

    # Ensure all columns from new_df exist in df
    combined_cols = list(dict.fromkeys(list(df.columns) + list(new_df.columns)))

    df_reidx = df.reindex(columns=combined_cols)
    new_df_reidx = new_df.reindex(columns=combined_cols)

    # Type-coerce values for columns that are numeric or boolean in the original df
    for col in combined_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                if col in new_df_reidx.columns:
                    new_df_reidx[col] = pd.to_numeric(new_df_reidx[col], errors='coerce')
            elif pd.api.types.is_bool_dtype(df[col].dtype):
                if col in new_df_reidx.columns:
                    new_df_reidx[col] = new_df_reidx[col].astype('boolean')
            else:
                if col in new_df_reidx.columns:
                    new_df_reidx[col] = new_df_reidx[col].astype(object)

    result = pd.concat([df_reidx, new_df_reidx], ignore_index=True, sort=False)
    return result


def apply_spec_overrides(overrides: Optional[Dict[str, Any]]) -> None:
    """Apply runtime overrides to POSITION_SPECS and queue specs from YAML/dict.
    
    Parameters
    ----------
    overrides : dict or None
        Structure:
        {
            'positions': {
                'baix': {'height_ratio_min': 1.0, 'height_weight': 0.0, ...},
                'crossa': {...},
                ...
            },
            'queues': {
                'mans': {'height_ratio_min': 0.52, 'height_weight': 0.8, ...},
                'daus': {...},
                'laterals': {...}
            }
        }
    """
    if overrides is None:
        return
    
    # Apply position overrides
    if 'positions' in overrides:
        for pos_name, params in overrides['positions'].items():
            if pos_name not in POSITION_SPECS:
                warnings.warn(f"Unknown position '{pos_name}' in overrides")
                continue
            
            spec = POSITION_SPECS[pos_name]
            for key, value in params.items():
                if hasattr(spec, key):
                    # Convert string enum values
                    if key == 'weight_preference' and isinstance(value, str):
                        value = WeightPreference(value.lower())
                    elif key == 'optimization_objective' and isinstance(value, str):
                        value = OptimizationObjective(value.lower())
                    setattr(spec, key, value)
                else:
                    warnings.warn(f"Unknown parameter '{key}' for position '{pos_name}'")
    
    # Apply queue overrides (shared across queue type)
    if 'queues' in overrides:
        queue_spec_maps = {
            'mans': MANS_QUEUE_SPECS,
            'daus': DAUS_QUEUE_SPECS,
            'laterals': LATERALS_QUEUE_SPECS
        }
        
        for queue_type, params in overrides['queues'].items():
            if queue_type not in queue_spec_maps:
                warnings.warn(f"Unknown queue type '{queue_type}' in overrides")
                continue
            
            # Apply to all queues of this type
            for queue_id, spec in queue_spec_maps[queue_type].items():
                for key, value in params.items():
                    if hasattr(spec, key):
                        if key == 'weight_preference' and isinstance(value, str):
                            value = WeightPreference(value.lower())
                        setattr(spec, key, value)
                    else:
                        warnings.warn(f"Unknown parameter '{key}' for queue type '{queue_type}'")
