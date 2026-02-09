import os
import yaml
from json import dump
from typing import Dict, List, Any, Optional
from soca.data import get_castellers, Casteller, add_castellers, apply_spec_overrides
from soca.assign import build_castell_assignment, apply_preassigned_to_all_assignments
from soca.utils import get_logger

logger = get_logger(__name__)

# Configuration file paths
CONFIG_YAML = 'config.yaml'
PREASSIGNED_YAML = 'preassigned.yaml'
UNTRACKED_YAML = 'untracked.yaml'


def load_yaml_file(filepath: str, required: bool = True) -> Optional[Dict[str, Any]]:
    """Load a YAML file with error handling.
    
    Parameters
    ----------
    filepath : str
        Path to YAML file
    required : bool
        If True, raise error if file doesn't exist. If False, return None.
    
    Returns
    -------
    dict or None
        Parsed YAML content, or None if file doesn't exist and not required
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist and required=True
    ValueError
        If YAML is invalid
    """
    if not os.path.exists(filepath):
        if required:
            raise FileNotFoundError(
                f"Required configuration file '{filepath}' not found."
            )
        else:
            logger.info("Optional file '%s' not found, skipping", filepath)
            return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {filepath}: {e}")


def parse_untracked_castellers(untracked_config: Optional[Dict[str, Any]]) -> List[Casteller]:
    """Parse untracked castellers from YAML config.
    
    Parameters
    ----------
    untracked_config : dict or None
        Content of untracked.yaml
    
    Returns
    -------
    list of Casteller
        Casteller instances to add to database
    """
    if untracked_config is None:
        return []
    
    castellers_data = untracked_config.get('castellers', [])
    
    castellers = []
    for item in castellers_data:
        if not isinstance(item, dict):
            logger.warning("Skipping invalid untracked casteller entry: %s", item)
            continue
        
        if not item.get('nom_complet'):
            logger.warning("Skipping untracked casteller without 'nom_complet': %s", item)
            continue
        
        # Create Casteller instance
        castellers.append(Casteller(
            nom_complet=item.get('nom_complet'),
            nom=item.get('nom'),
            cognoms=item.get('cognoms'),
            alcada_cm=item.get('alcada_cm'),
            alcada_espatlles_cm=item.get('alcada_espatlles_cm'),
            posicio_1=item.get('posicio_1', ''),
            posicio_2=item.get('posicio_2', '')
        ))
    
    return castellers


def parse_preassignments(preassigned_config: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, tuple]]:
    """Parse preassignments from YAML config.
    
    Parameters
    ----------
    preassigned_config : dict or None
        Content of preassigned.yaml
    
    Returns
    -------
    dict
        Preassignments in format: {position: {column: (name1, name2, ...)}}
    """
    if preassigned_config is None:
        return {}
    
    preassignments = {}
    
    for position_name, columns in preassigned_config.items():
        if not isinstance(columns, dict):
            logger.warning("Skipping invalid preassignment for position '%s'", position_name)
            continue
        
        preassignments[position_name] = {}
        for column_name, names in columns.items():
            if isinstance(names, list):
                # Filter out None/empty values
                valid_names = [n for n in names if n]
                if valid_names:
                    preassignments[position_name][column_name] = tuple(valid_names)
            elif isinstance(names, str) and names:
                # Single name as string
                preassignments[position_name][column_name] = (names,)
            else:
                logger.warning(
                    "Skipping invalid preassignment for %s[%s]: %s",
                    position_name, column_name, names
                )
    
    return preassignments


def parse_castell_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse castell structure configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from config.yaml
    
    Returns
    -------
    dict
        Castell configuration for build_castell_assignment
    """
    castell = config.get('castell', {})
    
    castell_config = {
        'columns': castell.get('columns', ['Rengla', 'Plena', 'Buida']),
        'tronc_positions': castell.get('tronc_positions', ['baix', 'segon', 'terç']),
        'include_mans': castell.get('include_mans', True),
        'include_daus': castell.get('include_daus', True),
        'include_laterals': castell.get('include_laterals', True),
        'include_crossa': castell.get('include_crossa', True),
        'include_contraforts': castell.get('include_contraforts', True),
        'include_agulles': castell.get('include_agulles', True),
    }
    
    return castell_config


def parse_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse optimization settings.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from config.yaml
    
    Returns
    -------
    dict
        Optimization settings
    """
    optimization = config.get('optimization', {})
    
    return {
        'method': optimization.get('method', 'adaptive_simulated_annealing'),
        'use_weight': optimization.get('use_weight', False),
    }


def main():
    """Main assignment pipeline with multi-file YAML configuration."""
    
    logger.info("="*60)
    logger.info("CASTELL ASSIGNMENT PIPELINE")
    logger.info("="*60)
    
    # Load main configuration (required)
    logger.info("\n[1/5] Loading main configuration")
    try:
        config = load_yaml_file(CONFIG_YAML, required=True)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Configuration error: %s", e)
        return
    
    # Load preassignments (optional)
    logger.info("[2/5] Loading preassignments")
    try:
        preassigned_config = load_yaml_file(PREASSIGNED_YAML, required=False)
        preassignments = parse_preassignments(preassigned_config)
        if preassignments:
            total_preassigned = sum(
                len(cols) for cols in preassignments.values()
            )
            logger.info("  Found %d preassignments across %d positions", 
                       total_preassigned, len(preassignments))
        else:
            logger.info("  No preassignments found")
    except ValueError as e:
        logger.error("Preassignment file error: %s", e)
        return
    
    # Load untracked castellers (optional)
    logger.info("[3/5] Loading untracked castellers")
    try:
        untracked_config = load_yaml_file(UNTRACKED_YAML, required=False)
        untracked = parse_untracked_castellers(untracked_config)
        if untracked:
            logger.info("  Found %d untracked castellers", len(untracked))
        else:
            logger.info("  No untracked castellers")
    except ValueError as e:
        logger.error("Untracked castellers file error: %s", e)
        return
    
    # Load casteller database
    logger.info("[4/5] Loading casteller database")
    castellers = get_castellers()
    logger.info("  Loaded %d castellers from database", len(castellers))
    
    # Add untracked castellers
    if untracked:
        castellers = add_castellers(castellers, untracked)
        logger.info("  Total castellers after adding untracked: %d", len(castellers))
    
    # Initialize assignments and apply preassignments
    all_assignments = {}
    if preassignments:
        logger.info("  Applying preassignments...")
        apply_preassigned_to_all_assignments(
            preassignments, 
            castellers, 
            all_assignments, 
            name_col='Nom complet'
        )
    
    # Parse castell configuration
    castell_config = parse_castell_config(config)
    logger.info("\n  Castell structure:")
    logger.info("    Columns: %s", ', '.join(castell_config['columns']))
    logger.info("    Tronc positions: %s", ', '.join(castell_config['tronc_positions']))
    logger.info("    Include mans: %s", castell_config['include_mans'])
    logger.info("    Include daus: %s", castell_config['include_daus'])
    logger.info("    Include laterals: %s", castell_config['include_laterals'])
    
    # Apply spec overrides (positions and queues)
    logger.info("\n  Applying position and queue specification overrides")
    overrides = {
        'positions': config.get('positions', {}),
        'queues': config.get('queues', {})
    }
    apply_spec_overrides(overrides)
    
    # Parse optimization settings
    optimization = parse_optimization_config(config)
    logger.info("\n  Optimization settings:")
    logger.info("    Method: %s", optimization['method'])
    logger.info("    Use weight: %s", optimization['use_weight'])
    
    # Run assignment
    logger.info("\n[5/5] Running castell assignment")
    logger.info("="*60)
    assignment = build_castell_assignment(
        castellers,
        castell_config,
        optimization_method=optimization['method'],
        use_weight=optimization['use_weight'],
        all_assignments=all_assignments
    )
    
    # Save output
    output_file = "assignment_output.json"
    with open(output_file, "w", encoding='utf-8') as f:
        dump(assignment, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Assignment complete!")
    logger.info("  Output saved to: %s", output_file)
    logger.info("="*60)


if __name__ == "__main__":
    main()