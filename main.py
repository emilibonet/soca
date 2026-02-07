import os
import yaml
from json import dump
from soca.data import get_castellers, Casteller, add_castellers, apply_spec_overrides
from soca.assign import build_castell_assignment, apply_preassigned_to_all_assignments

# Load casteller database
castellers = get_castellers()

untracked_castellers = [
    Casteller(nom_complet="Elena Gisbert", alcada_cm=162.0),
    Casteller(nom_complet="Nico Schmidt", alcada_cm=175.0),
    Casteller(nom_complet="Davide Mattei", alcada_cm=189.0),
    Casteller(nom_complet="Léa Braillon", alcada_cm=177.0),
    Casteller(nom_complet="Daria Ystayeva", alcada_cm=164.0),
    Casteller(nom_complet="Miquel Sistach", alcada_cm=185.0),
    Casteller(nom_complet="Sophie Aubert", alcada_cm=163.0),
    Casteller(nom_complet="Anna Romagosa", alcada_cm=170.0),
]


castellers = add_castellers(castellers, untracked_castellers)

preassigned = {
    "baix": {
        "Rengla": ("Mariona",),
        "Plena": ("Arnau",),
        # "Buida": ("Alexis",),
    },
    "segon": {
        "Rengla": ("Calvet",),
        "Plena": ("Vela",),
        "Buida": ("Emili",),
    },
}

# Initialize assignments and apply preassignments
all_assignments = {}
apply_preassigned_to_all_assignments(preassigned, castellers, all_assignments, name_col='Nom complet')

# Comprehensive castell configuration (use exact column names)
castell_config = {
    "columns": ["Rengla", "Plena", "Buida"],
    "tronc_positions": ["baix", "segon", "terç"],
    "mans": 3,
    "daus": 3,
    "laterals": 5,
    "include_mans": True,
    "include_daus": True,
    "include_laterals": True,
    "include_crossa": True,
    "include_contraforts": True,
    "include_agulles": True,
}

# Load config from YAML
CONFIG_YAML = 'config.yaml'
overrides = None
if os.path.exists(CONFIG_YAML):
    try:
        import yaml
        with open(CONFIG_YAML, 'r', encoding='utf-8') as yf:
            overrides = yaml.safe_load(yf) or {}
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load {CONFIG_YAML}: {e}")

# Apply overrides (YAML takes precedence over hardcoded defaults)
apply_spec_overrides(overrides)

# Run assignment
assignment = build_castell_assignment(
    castellers,
    castell_config,
    optimization_method='adaptive_simulated_annealing',
    use_weight=False,
    all_assignments=all_assignments
)

with open("assignment_output.json", "w") as f:
    dump(assignment, f, indent=2, ensure_ascii=False)