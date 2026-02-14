"""
Modern TUI display manager for castell assignment pipeline.
"""
import pandas as pd
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
import re
import time
import math
import sys

console = Console()

from .data import (
    MANS_QUEUE_SPECS,
    DAUS_QUEUE_SPECS,
    LATERALS_QUEUE_SPECS
)
from .optimize import (
    _calculate_reference_heights_for_queue,
    _calculate_candidate_score,
    create_position_spec_from_queue
)

def _interp(a, b, t):
    """Linear interpolation between a and b."""
    return int(a + (b - a) * t)


def _rgb_lerp(a, b, t):
    """Linear interpolation between two RGB tuples."""
    return (_interp(a[0], b[0], t),
            _interp(a[1], b[1], t),
            _interp(a[2], b[2], t))


def _colored_text(message: str, phase: float,
                  base=(100, 180, 200), peak=(150, 220, 255),
                  spread=4.0) -> Text:
    """Create Rich Text with RGB gradient animation.
    phase: 0..1 controls where the bright peak is positioned.
    spread: higher = sharper peak."""
    N = len(message)
    if N == 0:
        return Text()
    
    result = Text()
    for i, ch in enumerate(message):
        pos = i / max(1, N - 1)  # 0..1
        # circular distance between pos and phase
        d = abs(pos - phase)
        d = min(d, 1.0 - d)
        intensity = math.exp(-(d * spread) ** 2)  # 0..1
        r, g, b = _rgb_lerp(base, peak, intensity)
        result.append(ch, style=f"rgb({r},{g},{b})")
    
    return result


class SectionManager:
    """Manages TUI sections with active spinner and dimmed completed sections."""
    
    def __init__(self, accent_color: str = "cyan"):
        self.accent_color = accent_color
        self.base_rgb = self._parse_accent_color()
        self.peak_rgb = tuple(min(255, c + 50) for c in self.base_rgb)
        self.sections: List[Dict[str, Any]] = []
        self.current_section: Optional[str] = None
        self.live: Optional[Live] = None
        self.section_logs: Dict[str, List[str]] = {}
        self._animation_phase = 0.0
        self._stop_animation = False
        
    def _get_spinner_frames(self):
        """Return spinner animation frames."""
        return ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    def _parse_accent_color(self):
        """Convert accent_color to RGB tuple."""
        color = self.accent_color
        
        # Hex: #RRGGBB
        if color.startswith('#'):
            color = color.lstrip('#')
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        
        # RGB: rgb(r,g,b)
        if color.startswith('rgb'):
            match = re.findall(r'\d+', color)
            if len(match) == 3:
                return tuple(int(x) for x in match)
        
        # Named colors - common ones
        named = {
            'cyan': (0, 215, 255),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
        }
        return named.get(color, (100, 180, 200))  # fallback

    def _render_section(self, section: Dict[str, Any], is_current: bool) -> Text:
        """Render a single section header."""
        text = Text()
        
        if section['status'] == 'completed':
            # Completed - green checkmark, dim
            text.append("✓ ", style="green dim")
            text.append(section['title'], style="dim")
        elif section['status'] == 'failed':
            # Failed - red X, not dimmed
            text.append("✖ ", style="red bold")
            text.append(section['title'], style="red")
        elif is_current:
            # Current - animated spinner with gradient
            frame_idx = int(time.time() * 10) % len(self._get_spinner_frames())
            spinner = self._get_spinner_frames()[frame_idx]
            text.append(f"{spinner} ", style=f"bold {self.accent_color}")
            
            # Add gradient animation to title
            self._animation_phase = (time.time() * 0.5) % 1.0
            gradient_text = _colored_text(
                section['title'], self._animation_phase,
                base=self.base_rgb, peak=self.peak_rgb
            )
            text.append(gradient_text)
        else:
            # Pending - dimmed
            text.append("○ ", style="dim")
            text.append(section['title'], style="dim")
        
        return text
    
    def _render_logs(self, section_title: str) -> List[Text]:
        """Render logs for a section."""
        logs = self.section_logs.get(section_title, [])
        rendered = []
        
        for log in logs:
            # Parse log level
            if log.startswith('⚠'):
                style = "yellow"
            elif log.startswith('✓'):
                style = "green"
            elif log.startswith('ERROR') or log.startswith('CRITICAL'):
                style = "red bold"
            else:
                style = "white"
            
            rendered.append(Text(f"  {log}", style=style))
        
        return rendered
    
    def _render_display(self) -> Panel:
        """Render the complete display."""
        content = Text()
        
        for i, section in enumerate(self.sections):
            is_current = (section['title'] == self.current_section)
            
            # Add section header
            content.append(self._render_section(section, is_current))
            content.append("\n")
            
            # Show logs for ACTIVE sections, FAILED sections, or warnings from COMPLETED sections
            show_logs = (
                (is_current and section['status'] == 'active') or
                section['status'] == 'failed' or
                (section['status'] == 'completed' and self.section_logs.get(section['title'], []))
            )
            
            if show_logs:
                for log_line in self._render_logs(section['title']):
                    content.append(log_line)
                    content.append("\n")
            
            # Add spacing between sections (but not after last one)
            if i < len(self.sections) - 1:
                content.append("\n")
        
        return Panel(
            content,
            title="[bold]Castell Assignment Pipeline[/]",
            border_style=self.accent_color,
            padding=(1, 2)
        )
    
    def start(self):
        """Start the live display."""
        # Redirect stdout/stderr to suppress external prints
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        self.live = Live(
            self._render_display(),
            console=console,
            refresh_per_second=20,  # Higher refresh for smooth gradient
            transient=False  # Clear on stop
        )
        self.live.start()
        
        import threading
        def _animation_loop():
            import time
            while not self._stop_animation:
                time.sleep(1/30)  # 30fps
                try:
                    if self.live and not self._stop_animation:
                        self.live.update(self._render_display())
                except Exception:
                    break

        self._animation_thread = threading.Thread(target=_animation_loop, daemon=True)
        self._animation_thread.start()
    
    def stop(self):
        """Stop the live display."""
        self._stop_animation = True  # ADD
        if self.live:
            self.live.stop()

        # Restore stdout/stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
    
    def add_section(self, title: str):
        """Add a new section to the display."""
        self.sections.append({
            'title': title,
            'status': 'pending',
            'start_time': None,
            'end_time': None
        })
        self.section_logs[title] = []
    
    def activate_section(self, title: str):
        """Activate a section (show spinner and gradient)."""
        # Ensure section exists
        section_exists = any(s['title'] == title for s in self.sections)
        if not section_exists:
            self.add_section(title)
        
        self.current_section = title
        
        for section in self.sections:
            if section['title'] == title:
                section['status'] = 'active'
                section['start_time'] = time.time()
                break
        
        if self.live:
            self.live.update(self._render_display())
    
    def complete_section(self, title: str):
        """Mark a section as completed (show checkmark, clear logs except warnings)."""
        for section in self.sections:
            if section['title'] == title:
                section['status'] = 'completed'
                section['end_time'] = time.time()
                break
        
        # Keep only warnings for completed sections
        if title in self.section_logs:
            warnings = [log for log in self.section_logs[title] if log.startswith('⚠')]
            self.section_logs[title] = warnings
        
        if self.current_section == title:
            self.current_section = None
        
        if self.live:
            self.live.update(self._render_display())

    def fail_section(self, title: str):
        """Mark a section as failed."""
        for section in self.sections:
            if section['title'] == title:
                section['status'] = 'failed'
                section['end_time'] = time.time()
                break
        if self.current_section == title:
            self.current_section = None
        
        if self.live:
            self.live.update(self._render_display())

    def log(self, message: str, section: Optional[str] = None):
        """Add a log message to the current or specified section."""
        target_section = section or self.current_section
        
        if target_section and target_section in self.section_logs:
            self.section_logs[target_section].append(message)
            
            if self.live:
                self.live.update(self._render_display())
    
    @contextmanager
    def section(self, title: str):
        """Context manager for a section."""
        self.activate_section(title)
        try:
            yield self
        finally:
            for section in self.sections:
                if section['title'] == title and section['status'] != 'failed':
                    self.complete_section(title)
                    break


class SectionLogger:
    """Logger adapter that routes messages to section manager."""
    
    def __init__(self, section_manager: SectionManager, section_title: Optional[str] = None):
        self.manager = section_manager
        self.section_title = section_title
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        formatted = msg % args if args else msg
        self.manager.log(formatted, self.section_title)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        formatted = msg % args if args else msg
        self.manager.log(f"⚠ {formatted}", self.section_title)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        formatted = msg % args if args else msg
        self.manager.log(f"ERROR: {formatted}", self.section_title)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        formatted = msg % args if args else msg
        self.manager.log(f"CRITICAL: {formatted}", self.section_title)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        formatted = msg % args if args else msg
        self.manager.log(f"DEBUG: {formatted}", self.section_title)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception message."""
        formatted = msg % args if args else msg
        self.manager.log(f"EXCEPTION: {formatted}", self.section_title)


def _print_queue_summary(
    all_assignments: Dict,
    castellers: pd.DataFrame,
    columns: Dict[str, float],
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    peripheral_stats: Optional[Dict[str, Dict]] = None
):
    """Print formatted summary of queue assignments with reference/target info."""

    queue_mappings = {
        'mans': MANS_QUEUE_SPECS,
        'daus': DAUS_QUEUE_SPECS,
        'laterals': LATERALS_QUEUE_SPECS,
    }

    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type not in all_assignments:
            continue

        queue_specs = queue_mappings.get(queue_type, {})

        stats_for_queue = None
        if peripheral_stats:
            stats_for_queue = peripheral_stats.get(queue_type)

        if stats_for_queue and isinstance(stats_for_queue, dict) and 'final_score' in stats_for_queue:
            print(f"\n##### {queue_type.upper()} #####   score={stats_for_queue['final_score']:.2f}")
        else:
            print(f"\n##### {queue_type.upper()} #####")

        for queue_id, depth_list in all_assignments[queue_type].items():
            print(f"\n{queue_id}:")

            for depth_idx, assignment in enumerate(depth_list, start=1):
                # Compute reference and target where possible
                queue_spec = queue_specs.get(queue_id)
                ref_info = None
                if queue_spec is not None:
                    ref_height = _calculate_reference_heights_for_queue(
                        queue_spec, depth_idx, all_assignments, column_tronc_heights, castellers
                    )
                    
                    # Use spec ratios for depth 1, 0.80-1.00 for depth > 1
                    if depth_idx == 1:
                        min_ratio = queue_spec.height_ratio_min
                        max_ratio = queue_spec.height_ratio_max
                    else:
                        min_ratio = queue_spec.queue_height_ratio_min
                        max_ratio = queue_spec.queue_height_ratio_max
                    
                    min_target = ref_height * min_ratio
                    max_target = ref_height * max_ratio
                    ref_info = (ref_height, min_target, max_target)

                if assignment and assignment[0]:
                    name = assignment[0]
                    candidate_row = castellers[castellers['Nom complet'] == name].iloc[0]
                    h = candidate_row['Alçada (cm)']
                    
                    # Check expertise match
                    pos1 = str(candidate_row.get('Posició 1', '')).lower()
                    pos2 = str(candidate_row.get('Posició 2', '')).lower()
                    expertise_keywords = queue_spec.expertise_keywords if queue_spec else []
                    has_expertise = any(kw.lower() in pos1 or kw.lower() in pos2 
                                           for kw in expertise_keywords)
                    expertise_mark = "★" if has_expertise else " "
                    
                    # Compute per-person penalty (raw score scaled if queue stats provide factor)
                    try:
                        candidate_row = castellers[castellers['Nom complet'] == name].iloc[0]
                        pos_spec = None
                        if queue_spec is not None:
                            pos_spec = create_position_spec_from_queue(queue_spec, depth_idx)
                        if pos_spec is not None:
                            raw_score = _calculate_candidate_score(candidate_row, ref_info[0] if ref_info else h, pos_spec, [], True)
                        else:
                            raw_score = 0.0
                    except Exception:
                        raw_score = 0.0

                    if ref_info:
                        print(f"  {expertise_mark} Depth {depth_idx}: {name:<16} {h:3.0f} cm   ref={ref_info[0]:.1f} cm target={ref_info[1]:.1f}─{ref_info[2]:.1f} cm   penalty={raw_score:.2f}")
                    else:
                        print(f"  {expertise_mark} Depth {depth_idx}: {name:<16} {h:3.0f} cm   penalty={raw_score:.2f}")
                else:
                    empty_mark = "∅"
                    if ref_info:
                        print(f"  {empty_mark} Depth {depth_idx}: [empty]   ref={ref_info[0]:.1f} cm target={ref_info[1]:.1f}─{ref_info[2]:.1f} cm")
                    else:
                        print(f"  {empty_mark} Depth {depth_idx}: [empty]")


def _print_tronc_assignment(
    position_name: str,
    all_assignments: Dict,
    columns: Dict[str, float],
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    castellers: pd.DataFrame,
    position_stats: Optional[Dict] = None
):
    """Print tronc position assignments."""
    from .data import POSITION_SPECS
    
    if position_name not in POSITION_SPECS:
        return
    
    position_spec = POSITION_SPECS[position_name]
    
    # Calculate reference heights if needed; for positions without explicit
    # reference_positions (e.g., 'baix') use `columns` as reference heights.
    from .optimize import _calculate_reference_heights
    if position_spec.reference_positions:
        reference_heights = _calculate_reference_heights(
            columns, 
            position_spec.reference_positions,
            all_assignments,
            column_tronc_heights,
            castellers
        )
    else:
        # Use column base heights as the reference for positions like 'baix'
        reference_heights = {col: float(h) for col, h in columns.items()}
    
    print(f"\n##### {position_name.upper()} #####")
    if position_stats and isinstance(position_stats, dict) and 'final_score' in position_stats:
        print(f"  score={position_stats['final_score']:.2f}")
    
    for col_name, assigned in all_assignments[position_name].items():
        print(f"\n{col_name}:")
        
        if reference_heights and col_name in reference_heights:
            ref_height = reference_heights[col_name]
            min_target = ref_height * position_spec.height_ratio_min
            max_target = ref_height * position_spec.height_ratio_max
            print(f"  ref={ref_height:.1f} cm   target={min_target:.1f}─{max_target:.1f} cm")
        
        for i, name in enumerate(assigned, 1):
            if name:
                candidate_row = castellers[castellers['Nom complet'] == name].iloc[0]
                h = candidate_row['Alçada (cm)']
                
                # Check expertise match
                pos1 = str(candidate_row.get('Posició 1', '')).lower()
                pos2 = str(candidate_row.get('Posició 2', '')).lower()
                has_expertise = any(kw.lower() in pos1 or kw.lower() in pos2 
                                   for kw in position_spec.expertise_keywords)
                expertise_mark = "★" if has_expertise else " "
                
                # Compute per-person penalty using position spec and reference height
                try:
                    candidate_row = castellers[castellers['Nom complet'] == name].iloc[0]
                    ref_h = reference_heights[col_name] if reference_heights and col_name in reference_heights else h
                    raw_score = _calculate_candidate_score(candidate_row, ref_h, position_spec, [castellers[castellers['Nom complet'] == s].iloc[0] for s in assigned if s and s != name], True)
                except Exception:
                    raw_score = 0.0

                if reference_heights:
                    ratio_pct = (h / reference_heights[col_name]) * 100
                    print(f"  {expertise_mark} {name:<16} {h:3.0f} cm   {ratio_pct:5.1f}%   penalty={raw_score:.2f}")
                else:
                    print(f"  {name:<16} {h:3.0f} cm   penalty={raw_score:.2f}")


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


def summarize_assignments(
    all_assignments: Dict, 
    castellers: pd.DataFrame,
    columns: Dict[str, float],
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    assignment_stats: Dict[str, Dict] = None,
    peripheral_stats: Optional[Dict] = None
) -> Dict[str, int]:
    """Print comprehensive summary of all assignments."""
    
    # Print tronc position assignments first (structural order)
    for pos in ['baix', 'crossa', 'contrafort', 'agulla']:
        if pos in all_assignments:
            _print_tronc_assignment(
                pos, all_assignments, columns, column_tronc_heights, 
                castellers, assignment_stats.get(pos) if assignment_stats else None
            )

    # Then print queue summaries
    _print_queue_summary(all_assignments, castellers, columns, column_tronc_heights, peripheral_stats)
    
    # Validate
    errors, warnings = validate_structure(all_assignments, columns)
    
    if errors:
        print("="*60)
        print("CRITICAL STRUCTURAL ERRORS:")
        for err in errors:
            print(f"✗ {err}")
        print("="*60)
    
    if warnings:
        print("Warnings:")
        for warn in warnings:
            print(f"⚠ {warn}")
    
    # Print unassigned castellers
    try:
        all_assigned_names = set()
        
        for pos, assignments in all_assignments.items():
            if pos in ['mans', 'daus', 'laterals']:
                for queue_id, depth_list in assignments.items():
                    for assignment in depth_list:
                        if assignment and assignment[0]:
                            all_assigned_names.add(assignment[0])
            else:
                for col_assignments in assignments.values():
                    if col_assignments:
                        all_assigned_names.update(c for c in col_assignments if c)
        
        unassigned = castellers[~castellers['Nom complet'].isin(all_assigned_names)]
        if not unassigned.empty:
            print("\n##### UNASSIGNED CASTELLERS #####")
            for _, row in unassigned.iterrows():
                name = row.get('Nom complet', '')
                h = row.get('Alçada (cm)', None)
                pos1 = row.get('Posició 1', '')
                pos2 = row.get('Posició 2', '')
                expertise = ', '.join([p for p in [pos1, pos2] if p])
                height_str = f"{h:.1f} cm" if pd.notna(h) else "N/A"
                print(f"  - {name:<25} {height_str:8}  expertise={expertise}")
    except Exception as e:
        print(f"Failed to compute unassigned castellers list: {e}")
    
    return {'total_assigned': len(all_assigned_names), 'total_unassigned': len(unassigned)}


def create_final_panel(all_assignments: Dict, castellers, output_file: str) -> Panel:
    """Create final completion panel."""
    content = Text()
    
    # Count total assigned
    total_assigned = 0
    
    # Tronc positions
    for position in ['baix', 'crossa', 'contrafort', 'agulla']:
        if position in all_assignments:
            total_assigned += sum(
                len([p for p in assigned if p])
                for assigned in all_assignments[position].values()
            )
    
    # Queue positions
    for queue_type in ['mans', 'daus', 'laterals']:
        if queue_type in all_assignments:
            total_assigned += sum(
                len([d for d in depth_list if d and d[0]])
                for depth_list in all_assignments[queue_type].values()
            )
    
    content.append("✓ Assignment complete!\n\n", style="bold green")
    content.append(f"Total castellers assigned: ", style="white")
    content.append(f"{total_assigned}\n", style="bold cyan")
    content.append(f"Output saved to: ", style="white")
    content.append(f"{output_file}", style="bold cyan")
    
    return Panel(
        content,
        title="[bold green]Success[/]",
        border_style="green",
        padding=(1, 2)
    )