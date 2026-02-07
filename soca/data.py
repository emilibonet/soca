import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*include_tailing_empty.*",
    category=UserWarning
)   # Already dealing with empty headers.

import os
import pygsheets
import pandas as pd
from typing import List
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
    similarity_weight: float = 0.3
    weight_factor: float = 0.2
    
    # Optional constraints
    min_experience_level: int = 0  # 0=any, 1=secondary, 2=primary only

# Add new QueueSpec dataclass after WeightPreference
@dataclass
class QueueSpec:
    """Specification for a queue position (mans, daus, laterals)."""
    queue_id: str  # 'Rengla', 'R↔P', 'Rengla-left', etc.
    queue_type: str  # 'mans', 'daus', 'laterals'
    column_refs: List[str]  # ['Rengla'] or ['Rengla', 'Plena'] for daus
    expertise_keywords: List[str]
    height_ratio_min: float
    height_ratio_max: float
    weight_preference: WeightPreference = WeightPreference.NEUTRAL
    height_weight: float = 0.8
    expertise_weight: float = 0.5
    similarity_weight: float = 0.1
    weight_factor: float = 0.2

# Queue specifications for standard 3-column castell
MANS_QUEUE_SPECS = {
    'Rengla': QueueSpec(
        queue_id='Rengla',
        queue_type='mans',
        column_refs=['Rengla'],
        expertise_keywords=['primeres', 'general'],
        height_ratio_min=0.52,  # For depth 1
        height_ratio_max=1.0,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5
    ),
    'Plena': QueueSpec(
        queue_id='Plena',
        queue_type='mans',
        column_refs=['Plena'],
        expertise_keywords=['primeres', 'general'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5
    ),
    'Buida': QueueSpec(
        queue_id='Buida',
        queue_type='mans',
        column_refs=['Buida'],
        expertise_keywords=['primeres', 'general'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5
    ),
}

DAUS_QUEUE_SPECS = {
    'R↔P': QueueSpec(
        queue_id='R↔P',
        queue_type='daus',
        column_refs=['Rengla', 'Plena'],
        expertise_keywords=['Dau/Vent', 'dau', 'vent', 'dau/vent', 'general'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        weight_factor=0.3
    ),
    'P↔B': QueueSpec(
        queue_id='P↔B',
        queue_type='daus',
        column_refs=['Plena', 'Buida'],
        expertise_keywords=['Dau/Vent', 'dau', 'vent', 'dau/vent', 'general'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        weight_factor=0.3
    ),
    'B↔R': QueueSpec(
        queue_id='B↔R',
        queue_type='daus',
        column_refs=['Buida', 'Rengla'],
        expertise_keywords=['Dau/Vent', 'dau', 'vent', 'dau/vent', 'general'],
        height_ratio_min=0.52,
        height_ratio_max=1.0,
        weight_preference=WeightPreference.HEAVIER,
        height_weight=0.8,
        expertise_weight=0.5,
        weight_factor=0.3
    ),
}

LATERALS_QUEUE_SPECS = {
    'Rengla-left': QueueSpec(
        queue_id='Rengla-left',
        queue_type='laterals',
        column_refs=['Rengla'],
        expertise_keywords=['lateral', 'general'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5
    ),
    'Rengla-right': QueueSpec(
        queue_id='Rengla-right',
        queue_type='laterals',
        column_refs=['Rengla'],
        expertise_keywords=['lateral', 'general'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5
    ),
    'Plena-left': QueueSpec(
        queue_id='Plena-left',
        queue_type='laterals',
        column_refs=['Plena'],
        expertise_keywords=['lateral', 'general'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5
    ),
    'Plena-right': QueueSpec(
        queue_id='Plena-right',
        queue_type='laterals',
        column_refs=['Plena'],
        expertise_keywords=['lateral', 'general'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5
    ),
    'Buida-left': QueueSpec(
        queue_id='Buida-left',
        queue_type='laterals',
        column_refs=['Buida'],
        expertise_keywords=['lateral', 'general'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5
    ),
    'Buida-right': QueueSpec(
        queue_id='Buida-right',
        queue_type='laterals',
        column_refs=['Buida'],
        expertise_keywords=['lateral', 'general'],
        height_ratio_min=0.48,
        height_ratio_max=0.55,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5
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
        weight_factor=0.5
    ),
    
    'crossa': PositionRequirements(
        position_name='crossa',
        expertise_keywords=['crossa', 'cross'],
        count_per_column=2,
        reference_positions=['baix'],
        height_ratio_min=0.90,
        height_ratio_max=0.95,
        optimization_objective=OptimizationObjective.EVEN_DISTRIBUTION,
        weight_preference=WeightPreference.NEUTRAL,
        height_weight=1.0,
        expertise_weight=0.5,
        similarity_weight=0.5,
        weight_factor=0.2
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
        similarity_weight=0.3,
        weight_factor=0.3
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
        weight_factor=0.3
    ),
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
