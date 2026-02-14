#!/usr/bin/env python3
"""
Main assignment pipeline with TUI display integration.
"""
import os
import sys
import yaml
import argparse
from json import dump
from typing import Dict, List, Any, Optional
from rich.console import Console

# Add project root to path for display module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soca.data import get_castellers, Casteller, add_castellers, apply_spec_overrides, POSITION_SPECS
from soca.assign import apply_preassigned_to_all_assignments
from soca.optimize import find_optimal_assignment
from soca.utils import build_columns, compute_column_tronc_heights, filter_available_castellers
from soca.queue_assign import assign_rows_pipeline
from soca.display import SectionManager, SectionLogger, create_final_panel, summarize_assignments

# Configuration file paths
CONFIG_YAML = 'config.yaml'
PREASSIGNED_YAML = 'preassigned.yaml'
UNTRACKED_YAML = 'untracked.yaml'


def load_yaml_file(filepath: str, required: bool = True, logger=None) -> Optional[Dict[str, Any]]:
    """Load a YAML file with error handling.
    
    Parameters
    ----------
    filepath : str
        Path to YAML file
    required : bool
        If True, raise error if file doesn't exist. If False, return None.
    logger : SectionLogger
        Logger to use for messages
    
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
            if logger:
                logger.info(f"Optional file '{filepath}' not found, skipping")
            return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {filepath}: {e}")


def parse_untracked_castellers(untracked_config: Optional[Dict[str, Any]], logger=None) -> List[Casteller]:
    """Parse untracked castellers from YAML config.
    
    Parameters
    ----------
    untracked_config : dict or None
        Content of untracked.yaml
    logger : SectionLogger
        Logger for warnings
    
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
            if logger:
                logger.warning(f"Skipping invalid untracked casteller entry: {item}")
            continue
        
        if not item.get('nom_complet'):
            if logger:
                logger.warning(f"Skipping untracked casteller without 'nom_complet': {item}")
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


def parse_preassignments(preassigned_config: Optional[Dict[str, Any]], logger=None) -> Dict[str, Dict[str, tuple]]:
    """Parse preassignments from YAML config.
    
    Parameters
    ----------
    preassigned_config : dict or None
        Content of preassigned.yaml
    logger : SectionLogger
        Logger for warnings
    
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
            if logger:
                logger.warning(f"Skipping invalid preassignment for position '{position_name}'")
            continue
        
        preassignments[position_name] = {}
        for column_name, names in columns.items():
            # Handle nested lists (flatten for positions like crossa)
            if isinstance(names, list):
                # Check if it's a list of lists/strings (nested structure)
                flat_names = []
                for item in names:
                    if isinstance(item, list):
                        flat_names.extend([n for n in item if n])
                    elif isinstance(item, str) and item:
                        flat_names.append(item)
                
                if flat_names:
                    preassignments[position_name][column_name] = flat_names
                    
            elif isinstance(names, str) and names:
                # Single name as string
                preassignments[position_name][column_name] = [names]
            else:
                if logger:
                    logger.warning(
                        f"Skipping invalid preassignment for {position_name}[{column_name}]: {names}"
                    )
    
    return preassignments


def validate_preassignments(preassignments: Dict[str, Dict], logger=None) -> bool:
    """Validate preassignments for duplicates.
    
    Returns
    -------
    bool
        True if valid, False if duplicates found
    """
    # Collect all assigned names with their locations
    name_locations = {}
    
    for position_name, columns in preassignments.items():
        for column_name, names in columns.items():
            names_list = names if isinstance(names, (list, tuple)) else [names]
            
            for name in names_list:
                if not name:
                    continue
                    
                location = f"{position_name}[{column_name}]"
                if name in name_locations:
                    name_locations[name].append(location)
                else:
                    name_locations[name] = [location]
    
    # Find duplicates
    duplicates = {name: locs for name, locs in name_locations.items() if len(locs) > 1}
    
    if duplicates:
        if logger:
            logger.error("="*60)
            logger.error("ERROR: Duplicate preassignments detected")
            logger.error("="*60)
            for name, locations in sorted(duplicates.items()):
                logger.error(f"'{name}' appears {len(locations)} times:")
                for loc in locations:
                    logger.error(f"  - {loc}")
                logger.error("")
        else:
            print("="*60)
            print("ERROR: Duplicate preassignments detected")
            print("="*60)
            for name, locations in sorted(duplicates.items()):
                print(f"'{name}' appears {len(locations)} times:")
                for loc in locations:
                    print(f"  - {loc}")
                print("")
        return False
    
    return True


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


def build_summary_data(all_assignments: Dict, castellers, columns, column_tronc_heights) -> Dict[str, Any]:
    """Build comprehensive summary data for JSON output.
    
    Parameters
    ----------
    all_assignments : dict
        Complete assignment results
    castellers : DataFrame
        Casteller database
    columns : dict
        Column definitions
    column_tronc_heights : dict
        Computed tronc heights
    
    Returns
    -------
    dict
        Summary data including counts, penalties, unassigned, etc.
    """
    import pandas as pd
    from soca.data import POSITION_SPECS
    from soca.optimize import _calculate_candidate_score
    
    summary = {
        'total_assigned': 0,
        'positions': {},
        'queues': {},
        'unassigned': []
    }
    
    all_assigned_names = set()
    
    # Process tronc positions
    for position in ['baix', 'crossa', 'contrafort', 'agulla']:
        if position not in all_assignments:
            continue
        
        position_data = {
            'total_assigned': 0,
            'columns': {}
        }
        
        for col in columns.keys():
            if col not in all_assignments[position]:
                continue
            
            assigned = all_assignments[position][col]
            column_data = {
                'count': len([n for n in assigned if n]),
                'castellers': []
            }
            
            for name in assigned:
                if name:
                    all_assigned_names.add(name)
                    casteller = castellers[castellers['Nom complet'] == name]
                    
                    if not casteller.empty:
                        column_data['castellers'].append({
                            'name': name,
                            'height': float(casteller['Alçada (cm)'].iloc[0]),
                            'position_1': str(casteller.get('Posició 1', '').iloc[0]) if 'Posició 1' in casteller.columns else '',
                            'position_2': str(casteller.get('Posició 2', '').iloc[0]) if 'Posició 2' in casteller.columns else ''
                        })
            
            position_data['columns'][col] = column_data
            position_data['total_assigned'] += column_data['count']
        
        summary['positions'][position] = position_data
        summary['total_assigned'] += position_data['total_assigned']
    
    # Process queue positions
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type not in all_assignments:
            continue
        
        queue_data = {
            'total_assigned': 0,
            'queues': {}
        }
        
        for queue_id, depth_list in all_assignments[queue_type].items():
            queue_info = {
                'depths': []
            }
            
            for depth_idx, assignment in enumerate(depth_list, start=1):
                if assignment and assignment[0]:
                    name = assignment[0]
                    all_assigned_names.add(name)
                    casteller = castellers[castellers['Nom complet'] == name]
                    
                    if not casteller.empty:
                        queue_info['depths'].append({
                            'depth': depth_idx,
                            'name': name,
                            'height': float(casteller['Alçada (cm)'].iloc[0]),
                            'position_1': str(casteller.get('Posició 1', '').iloc[0]) if 'Posició 1' in casteller.columns else '',
                            'position_2': str(casteller.get('Posició 2', '').iloc[0]) if 'Posició 2' in casteller.columns else ''
                        })
                        queue_data['total_assigned'] += 1
            
            queue_data['queues'][queue_id] = queue_info
        
        summary['queues'][queue_type] = queue_data
        summary['total_assigned'] += queue_data['total_assigned']
    
    # Unassigned castellers
    unassigned = castellers[~castellers['Nom complet'].isin(all_assigned_names)]
    for _, row in unassigned.iterrows():
        summary['unassigned'].append({
            'name': row.get('Nom complet', ''),
            'height': float(row.get('Alçada (cm)', 0)) if pd.notna(row.get('Alçada (cm)')) else None,
            'position_1': str(row.get('Posició 1', '')),
            'position_2': str(row.get('Posició 2', ''))
        })
    
    return summary


def assign_single_position(
    position_name: str,
    castellers,
    columns,
    all_assignments: Dict,
    column_tronc_heights,
    optimization_method: str,
    use_weight: bool,
    logger: SectionLogger
) -> Dict[str, Any]:
    """Assign a single position with logging.
    
    Parameters
    ----------
    position_name : str
        Name of position to assign (e.g., 'baix', 'crossa')
    castellers : DataFrame
        Casteller database
    columns : dict
        Column definitions
    all_assignments : dict
        Current assignments
    column_tronc_heights : dict or None
        Tronc heights (None for pinya-level)
    optimization_method : str
        Optimization method to use
    use_weight : bool
        Whether to use weight in optimization
    logger : SectionLogger
        Logger for this section
    
    Returns
    -------
    dict
        Assignment statistics
    """
    # Check if position already fully assigned
    already_assigned_columns = set(all_assignments.get(position_name, {}).keys())
    missing_columns = [c for c in columns.keys() 
                      if c not in already_assigned_columns 
                      or not all_assignments.get(position_name, {}).get(c)]
    
    if not missing_columns:
        logger.info(f"✓ {position_name.upper()} already fully assigned")
        return {}
    
    # Check if spec exists
    if position_name not in POSITION_SPECS:
        logger.error(f"Missing POSITION_SPECS entry for '{position_name}'")
        raise ValueError(f"Missing POSITION_SPECS entry for '{position_name}'")
    
    logger.info(f"Starting {position_name.upper()} assignment...")
    
    # Count available candidates
    available = filter_available_castellers(castellers, all_assignments)
    logger.info(f"Available candidates: {len(available)}")
    
    # Get position spec
    spec = POSITION_SPECS[position_name]
    logger.info(f"Optimization method: {optimization_method}")
    
    # Run optimization
    logger.info("Initializing optimization...")
    computed_assignment, stats = find_optimal_assignment(
        castellers=castellers,
        position_spec=spec,
        previous_assignments=all_assignments,
        columns=columns,
        column_tronc_heights=column_tronc_heights,
        optimization_method=optimization_method,
        use_weight=use_weight,
        return_stats=True
    )
    
    # Update assignments
    all_assignments.setdefault(position_name, {})
    for col_name, value in computed_assignment.items():
        if col_name not in all_assignments[position_name] or not all_assignments[position_name].get(col_name):
            all_assignments[position_name][col_name] = value
    
    # Log results
    total_assigned = sum(len([n for n in assigned if n]) 
                        for assigned in computed_assignment.values())
    
    if stats and 'final_score' in stats:
        logger.info(f"✓ {position_name.upper()} assignment complete (score: {stats['final_score']:.1f})")
    else:
        logger.info(f"✓ {position_name.upper()} assignment complete ({total_assigned} castellers)")
    
    return stats


def main():
    """Main assignment pipeline with TUI display."""
    # ===================================================================
    # LOAD CONFIG FOR TUI SETUP
    # ===================================================================
    # Parse CLI args for optional overrides (e.g. custom preassigned file)
    parser = argparse.ArgumentParser(description="SOCA assignment runner")
    parser.add_argument(
        "--preassigned", "-p",
        help="Path to preassigned YAML file (overrides default)",
        default=PREASSIGNED_YAML
    )
    args = parser.parse_args()
    preassigned_path = args.preassigned or PREASSIGNED_YAML

    try:
        with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
    
    accent_color = config.get('display', {}).get('accent_color', 'cyan')
    
    # ===================================================================
    # SETUP TUI MANAGER
    # ===================================================================
    tui = SectionManager(accent_color=accent_color)
    # Set TUI manager for all modules
    import soca.optimize
    import soca.queue_assign
    soca.optimize._tui_manager = tui
    soca.queue_assign._tui_manager = tui
    # Define all sections upfront
    tui.add_section("Loading configuration")
    tui.add_section("Loading data")
    tui.add_section("Assigning BAIX")
    tui.add_section("Assigning CROSSA")
    tui.add_section("Assigning CONTRAFORT")
    tui.add_section("Assigning AGULLA")
    tui.add_section("Assigning peripheral positions")
    
    # Start the TUI (this captures stdout/stderr)
    tui.start()
    
    try:
        # ================================================================
        # SECTION 1: LOADING CONFIGURATION
        # ================================================================
        with tui.section("Loading configuration"):
            logger = SectionLogger(tui, "Loading configuration")
            
            try:
                config = load_yaml_file(CONFIG_YAML, required=True, logger=logger)
                logger.info("[1/5] Main configuration loaded")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Configuration error: {e}")
                return
            
            # Load preassignments
            logger.info("[2/5] Checking for preassignments...")
            preassigned_config = load_yaml_file(preassigned_path, required=False, logger=logger)
            preassignments = parse_preassignments(preassigned_config, logger=logger)
            if preassignments:
                total_preassigned = sum(len(cols) for cols in preassignments.values())
                logger.info(f"[2/5] Found {total_preassigned} preassignments across {len(preassignments)} positions")
            else:
                logger.info("[2/5] No preassignments found")
            
            # Load untracked castellers
            logger.info("[3/5] Checking for untracked castellers...")
            untracked_config = load_yaml_file(UNTRACKED_YAML, required=False, logger=logger)
            untracked = parse_untracked_castellers(untracked_config, logger=logger)
            if untracked:
                logger.info(f"[3/5] Found {len(untracked)} untracked castellers")
            else:
                logger.info("[3/5] No untracked castellers")
        
        # ================================================================
        # SECTION 2: LOADING DATA
        # ================================================================
        with tui.section("Loading data"):
            logger = SectionLogger(tui, "Loading data")
            
            # Load casteller database
            logger.info("[4/5] Loading castellers from database...")
            castellers = get_castellers()
            logger.info(f"[4/5] Loaded {len(castellers)} castellers from database")
            
            # Add untracked castellers
            if untracked:
                logger.info("Adding untracked castellers...")
                castellers = add_castellers(castellers, untracked)
                logger.info(f"Total castellers after adding untracked: {len(castellers)}")
            
            # Initialize assignments and apply preassignments
            all_assignments = {}
            if preassignments:
                # Validate for duplicates
                if not validate_preassignments(preassignments, logger):
                    tui.fail_section("Loading data")
                    raise ValueError("Preassignment validation failed: duplicate assignments detected")
                
                logger.info("Applying preassignments...")
                apply_preassigned_to_all_assignments(
                    preassignments, 
                    castellers, 
                    all_assignments, 
                    name_col='Nom complet',
                    logger_override=logger
                )
                logger.info("✓ Preassignments applied")
            
            # Parse castell configuration
            castell_config = parse_castell_config(config)
            logger.info(f"Columns: {', '.join(castell_config['columns'])}")
            logger.info(f"Tronc positions: {', '.join(castell_config['tronc_positions'])}")
            
            # Apply spec overrides
            overrides = {
                'positions': config.get('positions', {}),
                'queues': config.get('queues', {})
            }
            apply_spec_overrides(overrides)
            
            # Parse optimization settings
            optimization = parse_optimization_config(config)
            logger.info(f"[5/5] Optimization method: {optimization['method']}")
        
        # Build columns
        columns = build_columns(castell_config['columns'])
        
        # Initialize position assignments
        for pos in castell_config['tronc_positions']:
            all_assignments.setdefault(pos, {})
        
        # ================================================================
        # SECTION 3: ASSIGNING BAIX
        # ================================================================
        with tui.section("Assigning BAIX"):
            logger = SectionLogger(tui, "Assigning BAIX")
            try:
                assign_single_position(
                    position_name='baix',
                    castellers=castellers,
                    columns=columns,
                    all_assignments=all_assignments,
                    column_tronc_heights=None,  # Pinya level
                    optimization_method=optimization['method'],
                    use_weight=optimization['use_weight'],
                    logger=logger
                )
            except Exception as e:
                tui.fail_section("Assigning BAIX")
                raise
        
        # ================================================================
        # SECTION 4: ASSIGNING CROSSA
        # ================================================================
        if castell_config.get('include_crossa', True):
            with tui.section("Assigning CROSSA"):
                logger = SectionLogger(tui, "Assigning CROSSA")
                try:
                    assign_single_position(
                        position_name='crossa',
                        castellers=castellers,
                        columns=columns,
                        all_assignments=all_assignments,
                        column_tronc_heights=None,  # Pinya level
                        optimization_method=optimization['method'],
                        use_weight=optimization['use_weight'],
                        logger=logger
                    )
                except Exception as e:
                    tui.fail_section("Assigning BAIX")
                    raise
        
        # ================================================================
        # SECTION 5: ASSIGNING CONTRAFORT
        # ================================================================
        if castell_config.get('include_contraforts', True):
            with tui.section("Assigning CONTRAFORT"):
                logger = SectionLogger(tui, "Assigning CONTRAFORT")
                try:
                    assign_single_position(
                        position_name='contrafort',
                        castellers=castellers,
                        columns=columns,
                        all_assignments=all_assignments,
                        column_tronc_heights=None,  # Pinya level
                        optimization_method=optimization['method'],
                        use_weight=optimization['use_weight'],
                        logger=logger
                    )
                except Exception as e:
                    tui.fail_section("Assigning CONTRAFORT")
                    raise
        
        # Compute tronc heights after pinya positions
        column_tronc_heights = compute_column_tronc_heights(
            all_assignments,
            castell_config['tronc_positions'],
            castellers
        )
        
        # ================================================================
        # SECTION 6: ASSIGNING AGULLA
        # ================================================================
        if castell_config.get('include_agulles', True):
            with tui.section("Assigning AGULLA"):
                logger = SectionLogger(tui, "Assigning AGULLA")
                
                logger.info("Computing tronc reference heights...")
                try:
                    assign_single_position(
                        position_name='agulla',
                        castellers=castellers,
                        columns=columns,
                        all_assignments=all_assignments,
                        column_tronc_heights=column_tronc_heights,
                        optimization_method=optimization['method'],
                        use_weight=optimization['use_weight'],
                        logger=logger
                    )
                except Exception as e:
                    tui.fail_section("Assigning AGULLA")
                    raise
        
        # ================================================================
        # SECTION 7: ASSIGNING PERIPHERAL POSITIONS
        # ================================================================
        with tui.section("Assigning peripheral positions"):
            logger = SectionLogger(tui, "Assigning peripheral positions")
            
            logger.info("Starting peripheral assignment (mans, daus, laterals)...")
            logger.info("Building queue specifications...")
            try:
                result, stats = assign_rows_pipeline(
                    castellers=castellers,
                    columns=columns,
                    column_tronc_heights=column_tronc_heights,
                    all_assignments=all_assignments,
                    include_mans=castell_config.get('include_mans', True),
                    include_daus=castell_config.get('include_daus', True),
                    include_laterals=castell_config.get('include_laterals', True)
                )
            except Exception as e:
                tui.fail_section("Assigning peripheral positions")
                raise
            
            for queue_type in ['mans', 'daus', 'laterals']:
                if queue_type in result:
                    all_assignments[queue_type] = result[queue_type]
            
            # Log summary per queue type
            for queue_type in ['mans', 'daus', 'laterals']:
                if queue_type in result:
                    logger.info(f"{queue_type.upper()} assigned:")
                    for queue_id, depth_list in result[queue_type].items():
                        filled = len([d for d in depth_list if d and d[0]])
                        logger.info(f"  {queue_id}: {filled} filled")
            
            if stats and 'total_score' in stats:
                logger.info(f"✓ Peripheral assignment complete (global score: {stats['total_score']:.1f})")
            else:
                logger.info("✓ Peripheral assignment complete")
        
        # ================================================================
        # BUILD SUMMARY AND SAVE OUTPUT
        # ================================================================
        summary_data = build_summary_data(
            all_assignments,
            castellers,
            columns,
            column_tronc_heights
        )
        
        # Save output with summary
        output_file = "assignment_output.json"
        output_data = {
            'assignments': all_assignments,
            'summary': summary_data,
            'configuration': {
                'columns': castell_config['columns'],
                'tronc_positions': castell_config['tronc_positions'],
                'optimization_method': optimization['method'],
                'use_weight': optimization['use_weight']
            }
        }
        
        with open(output_file, "w", encoding='utf-8') as f:
            dump(output_data, f, indent=2, ensure_ascii=False)
        
        console = Console(file=sys.stdout, force_terminal=True)
        
        # Print detailed assignment summary
        console.print("\n" + "="*60)
        console.print("FINAL ASSIGNMENTS")
        console.print("="*60 + "\n")

        summary = summarize_assignments(
            all_assignments=all_assignments,
            castellers=castellers,
            columns=columns,
            column_tronc_heights=column_tronc_heights,
            assignment_stats={},
            peripheral_stats=stats
        )
        console.print()
        console.print(create_final_panel(all_assignments, castellers, output_file))
        console.print()
    
    finally:        
        tui.stop()


if __name__ == "__main__":
    main()