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
import io
import math
import sys
import threading

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


# Style-string cache keyed on (r, g, b) — avoids repeated f-string allocation
_STYLE_CACHE: Dict[Tuple[int, int, int], str] = {}


def _rgb_style(r: int, g: int, b: int) -> str:
    key = (r, g, b)
    s = _STYLE_CACHE.get(key)
    if s is None:
        s = f"rgb({r},{g},{b})"
        _STYLE_CACHE[key] = s
    return s


def _colored_text(message: str, phase: float,
                  base=(100, 180, 200), peak=(150, 220, 255),
                  spread=4.0) -> Text:
    """Create Rich Text with RGB gradient animation.
    phase: 0..1 controls where the bright peak is positioned.
    spread: higher = sharper peak."""
    N = len(message)
    if N == 0:
        return Text()

    # Hoist lookups to locals
    _exp = math.exp
    _lerp = _rgb_lerp
    _style = _rgb_style
    spread_sq = spread * spread
    inv_n = 1.0 / max(1, N - 1)

    # Build runs of consecutive characters sharing the same style
    result = Text()
    prev_st: Optional[str] = None
    buf: List[str] = []

    for i in range(N):
        pos = i * inv_n
        d = abs(pos - phase)
        if d > 0.5:
            d = 1.0 - d
        intensity = _exp(-(d * d * spread_sq))
        r, g, b = _lerp(base, peak, intensity)
        st = _style(r, g, b)
        if st == prev_st:
            buf.append(message[i])
        else:
            if buf:
                result.append("".join(buf), style=prev_st)
            buf = [message[i]]
            prev_st = st
    if buf:
        result.append("".join(buf), style=prev_st)

    return result


class SectionManager:
    """Manages TUI sections with active spinner and dimmed completed sections."""
    
    def __init__(
        self,
        accent_color: str = "cyan",
        animation_enabled: bool = True,
        panel_title: str = "Castell Assignment Pipeline",
        transient: bool = False,
        refresh_per_second: int = 4,
    ):
        self.accent_color = accent_color
        self.animation_enabled = animation_enabled
        self.panel_title = panel_title
        self.transient = transient
        self.refresh_per_second = max(1, refresh_per_second)
        self.base_rgb = self._parse_accent_color()
        self.peak_rgb = tuple(min(255, c + 50) for c in self.base_rgb)
        self.sections: List[Dict[str, Any]] = []
        self.current_section: Optional[str] = None
        self.live: Optional[Live] = None
        self.section_logs: Dict[str, List[str]] = {}
        self._animation_phase = 0.0
        self._stop_animation = False
        self._scroll_offset = 0
        self._needs_update = False
        self._last_render_time = 0
        self._min_render_interval = 1.0 / self.refresh_per_second
        self._cached_display: Optional[Panel] = None
        self._last_section_count = 0
        self._last_log_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def _get_spinner_frames(self):
        """Return spinner animation frames."""
        return ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    def _parse_accent_color(self):
        """Convert accent_color to RGB tuple."""
        color = self.accent_color
        
        # Hex: #RGB or #RRGGBB
        if color.startswith('#'):
            hex_str = color[1:]  # strip exactly one '#'
            if len(hex_str) == 3:
                # Expand short form: #FAB → #FFAABB
                hex_str = hex_str[0]*2 + hex_str[1]*2 + hex_str[2]*2
            if len(hex_str) != 6 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
                return (100, 180, 200)  # fallback for malformed hex
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        
        # RGB: rgb(r,g,b)
        if color.startswith('rgb'):
            match = re.findall(r'\d+', color)
            if len(match) == 3:
                return tuple(min(255, max(0, int(x))) for x in match)
        
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
            if self.animation_enabled:
                frame_idx = int(time.time() * 10) % len(self._get_spinner_frames())
                spinner = self._get_spinner_frames()[frame_idx]
                text.append(f"{spinner} ", style=f"bold {self.accent_color}")
                
                # Add gradient animation to title (2s sweep + 1s pause)
                cycle = (time.time() % 3.0) / 3.0   # 0..1 over 3 seconds
                # First 2/3 of the cycle: sweep (phase 0→1)
                # Last  1/3 of the cycle: hold at base (phase parked off-screen)
                if cycle < 2.0 / 3.0:
                    self._animation_phase = cycle * 1.5   # 0→1 in 2s
                else:
                    self._animation_phase = -1.0           # off-screen → all base
                gradient_text = _colored_text(
                    section['title'], self._animation_phase,
                    base=self.base_rgb, peak=self.peak_rgb
                )
                text.append(gradient_text)
            else:
                text.append("● ", style=f"bold {self.accent_color}")
                text.append(section['title'], style=f"{self.accent_color}")
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
    
    def _build_all_lines(self) -> List[Text]:
        """Build all renderable lines (headers + logs + spacing)."""
        lines: List[Text] = []

        for i, section in enumerate(self.sections):
            is_current = (section['title'] == self.current_section)
            lines.append(self._render_section(section, is_current))

            show_logs = (
                (is_current and section['status'] == 'active') or
                section['status'] == 'failed' or
                (section['status'] == 'completed'
                 and self.section_logs.get(section['title'], []))
            )
            if show_logs:
                lines.extend(self._render_logs(section['title']))

            # Spacing between sections (not after the last one)
            if i < len(self.sections) - 1:
                lines.append(Text(""))

        return lines

    def _render_display(self) -> Panel:
        """Render the display, bounded to terminal height with auto-scroll."""
        all_lines = self._build_all_lines()
        total = len(all_lines)

        # Panel overhead: top border, padding-top, padding-bottom, bottom border
        # = 4 lines.  Title occupies the top border line so no extra cost.
        panel_overhead = 4
        terminal_h = console.height or 24
        available = max(3, terminal_h - panel_overhead)

        if total <= available:
            # Everything fits — render normally
            visible = all_lines
            truncated_above = 0
        else:
            # Determine view window
            if self._scroll_offset == 0:
                # Auto-follow: show the last `available` lines
                start = total - available
            else:
                start = max(0, total - available - self._scroll_offset)
            end = start + available
            visible = all_lines[start:end]
            truncated_above = start

        content = Text()
        if truncated_above > 0:
            hint = Text(
                f"  ↑ {truncated_above} more line"
                f"{'s' if truncated_above != 1 else ''} above",
                style="dim italic",
            )
            content.append(hint)
            content.append("\n")

        for idx, line in enumerate(visible):
            content.append(line)
            if idx < len(visible) - 1:
                content.append("\n")

        # Show hint when scrolled up and there are lines below
        if self._scroll_offset > 0 and total > available:
            start = max(0, total - available - self._scroll_offset)
            actual_below = max(0, total - (start + available))
            if actual_below > 0:
                content.append("\n")
                content.append(
                    Text(
                        f"  ↓ {actual_below} more line"
                        f"{'s' if actual_below != 1 else ''} below",
                        style="dim italic",
                    )
                )

        return Panel(
            content,
            title=f"[bold]{self.panel_title}[/]",
            border_style=self.accent_color,
            padding=(1, 2),
        )

    def scroll_up(self, lines: int = 3):
        """Scroll up (towards older content)."""
        with self._lock:
            all_lines = self._build_all_lines()
            terminal_h = console.height or 24
            available = max(3, terminal_h - 4)
            max_offset = max(0, len(all_lines) - available)
            self._scroll_offset = min(max_offset, self._scroll_offset + lines)
            self._needs_update = True

    def scroll_down(self, lines: int = 3):
        """Scroll down (towards newer content). 0 = auto-follow."""
        with self._lock:
            self._scroll_offset = max(0, self._scroll_offset - lines)
            self._needs_update = True
    
    def start(self):
        """Start the live display."""
        self.live = Live(
            self._render_display(),
            console=console,
            refresh_per_second=self.refresh_per_second,
            transient=self.transient,
        )
        self.live.start()
        
        def _animation_loop():
            """Highly optimized animation loop - minimal updates."""
            while not self._stop_animation:
                current_time = time.time()
                
                # Only update animation if there's an active section
                with self._lock:
                    if self.current_section and self.animation_enabled:
                        # Check if enough time has passed
                        if current_time - self._last_render_time >= self._min_render_interval:
                            self._needs_update = True
                            self._last_render_time = current_time
                    
                    should_update = self._needs_update
                
                # Only render if something actually changed
                if should_update:
                    try:
                        if self.live and not self._stop_animation:
                            self.live.update(self._render_display())
                            with self._lock:
                                self._needs_update = False
                    except Exception:
                        break
                
                # Longer sleep when idle or animations disabled to save CPU
                if not self.animation_enabled:
                    sleep_time = 0.5
                elif self.current_section:
                    sleep_time = self._min_render_interval
                else:
                    sleep_time = 0.5
                time.sleep(sleep_time)

        self._animation_thread = threading.Thread(target=_animation_loop, daemon=True)
        self._animation_thread.start()

        # Keyboard listener for scroll (non-blocking stdin reader)
        def _key_loop():
            import tty, termios, select
            fd = sys.stdin.fileno()
            try:
                old_settings = termios.tcgetattr(fd)
            except termios.error:
                return  # not a real terminal
            try:
                tty.setcbreak(fd)
                while not self._stop_animation:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        if ch == '\x1b':  # escape sequence
                            if select.select([sys.stdin], [], [], 0.05)[0]:
                                ch2 = sys.stdin.read(1)
                                if ch2 == '[':
                                    if select.select([sys.stdin], [], [], 0.05)[0]:
                                        ch3 = sys.stdin.read(1)
                                        if ch3 == 'A':  # Up arrow
                                            self.scroll_up(1)
                                        elif ch3 == 'B':  # Down arrow
                                            self.scroll_down(1)
                                        elif ch3 == '5':  # Page Up
                                            sys.stdin.read(1)  # consume ~
                                            self.scroll_up(10)
                                        elif ch3 == '6':  # Page Down
                                            sys.stdin.read(1)  # consume ~
                                            self.scroll_down(10)
                        elif ch in ('k', 'K'):
                            self.scroll_up(1)
                        elif ch in ('j', 'J'):
                            self.scroll_down(1)
                        elif ch in ('g', 'G'):
                            # G = bottom (auto-follow), g = top
                            if ch == 'g':
                                with self._lock:
                                    all_lines = self._build_all_lines()
                                    terminal_h = console.height or 24
                                    available = max(3, terminal_h - 4)
                                    self._scroll_offset = max(0, len(all_lines) - available)
                                    self._needs_update = True
                            else:
                                with self._lock:
                                    self._scroll_offset = 0
                                    self._needs_update = True
            except Exception:
                pass
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

        self._key_thread = threading.Thread(target=_key_loop, daemon=True)
        self._key_thread.start()
    
    def stop(self):
        """Stop the live display."""
        self._stop_animation = True
        if self.live:
            self.live.stop()
    
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
        
        with self._lock:
            self.current_section = title
        
        for section in self.sections:
            if section['title'] == title:
                section['status'] = 'active'
                section['start_time'] = time.time()
                break
        
        self._needs_update = True
    
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
        
        with self._lock:
            if self.current_section == title:
                self.current_section = None
            self._needs_update = True

    def fail_section(self, title: str):
        """Mark a section as failed."""
        for section in self.sections:
            if section['title'] == title:
                section['status'] = 'failed'
                section['end_time'] = time.time()
                break
        with self._lock:
            if self.current_section == title:
                self.current_section = None
            self._needs_update = True

    def log(self, message: str, section: Optional[str] = None):
        """Add a log message to the current or specified section."""
        target_section = section or self.current_section
        
        if target_section and target_section in self.section_logs:
            with self._lock:
                self.section_logs[target_section].append(message)
                
                # Reset scroll to auto-follow when new log arrives
                self._scroll_offset = 0
                
                self._needs_update = True
    
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
    peripheral_stats: Optional[Dict[str, Dict]] = None,
    preassigned_names: Optional[set] = None,
    preassigned_swapped_names: Optional[set] = None,
    failed_preassignments: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None,
):
    """Generate formatted summary of queue assignments with reference/target info."""
    if preassigned_names is None:
        preassigned_names = set()
    if preassigned_swapped_names is None:
        preassigned_swapped_names = set()
    if failed_preassignments is None:
        failed_preassignments = {}

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
            yield f"\n##### {queue_type.upper()} #####   score={stats_for_queue['final_score']:.2f}\n"
        else:
            yield f"\n##### {queue_type.upper()} #####\n"

        for queue_id, depth_list in all_assignments[queue_type].items():
            yield f"\n{queue_id}:\n"

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

                    # Determine replacement info for this slot
                    replaced_name = failed_preassignments.get(queue_type, {}).get(queue_id, {}).get(depth_idx - 1)
                    replaces_note = ""
                    is_replacement = replaced_name is not None and name not in preassigned_names
                    if is_replacement:
                        replaces_note = f"  ─ replaces {replaced_name}"

                    # Pin mark: ⌖ = preassigned here, ⇆ = replaces a failed preassignment
                    if name in preassigned_names:
                        pin_mark = "⌖ "
                    elif is_replacement:
                        pin_mark = "⇆ "
                    else:
                        pin_mark = "  "
                    
                    # Compute per-person penalty (raw score scaled if queue stats provide factor)
                    try:
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
                        yield f"{pin_mark}{expertise_mark} Depth {depth_idx}: {name:<16} {h:3.0f} cm   ref={ref_info[0]:.1f} cm target={ref_info[1]:.1f}─{ref_info[2]:.1f} cm   penalty={raw_score:.2f}{replaces_note}\n"
                    else:
                        yield f"{pin_mark}{expertise_mark} Depth {depth_idx}: {name:<16} {h:3.0f} cm   penalty={raw_score:.2f}{replaces_note}\n"
                else:
                    empty_mark = "∅"
                    missing_name = failed_preassignments.get(queue_type, {}).get(queue_id, {}).get(depth_idx - 1)
                    missing_note = f"             ─ missing {missing_name}" if missing_name else ""
                    if ref_info:
                        yield f"  {empty_mark} Depth {depth_idx}: [empty]   ref={ref_info[0]:.1f} cm target={ref_info[1]:.1f}─{ref_info[2]:.1f} cm{missing_note}\n"
                    else:
                        yield f"  {empty_mark} Depth {depth_idx}: [empty]{missing_note}\n"


def _print_tronc_assignment(
    position_name: str,
    all_assignments: Dict,
    columns: Dict[str, float],
    column_tronc_heights: Optional[Dict[str, Dict[str, float]]],
    castellers: pd.DataFrame,
    position_stats: Optional[Dict] = None,
    preassigned_names: Optional[set] = None,
    preassigned_swapped_names: Optional[set] = None,
    failed_preassignments: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None,
):
    """Generate tronc position assignments."""
    if preassigned_names is None:
        preassigned_names = set()
    if preassigned_swapped_names is None:
        preassigned_swapped_names = set()
    if failed_preassignments is None:
        failed_preassignments = {}
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
    
    yield f"\n##### {position_name.upper()} #####\n"
    if position_stats and isinstance(position_stats, dict) and 'final_score' in position_stats:
        yield f"  score={position_stats['final_score']:.2f}\n"
    
    for col_name, assigned in all_assignments[position_name].items():
        yield f"\n{col_name}:\n"
        if reference_heights and col_name in reference_heights:
            ref_height = reference_heights[col_name]
            min_target = ref_height * position_spec.height_ratio_min
            max_target = ref_height * position_spec.height_ratio_max
            yield f"  ref={ref_height:.1f} cm   target={min_target:.1f}─{max_target:.1f} cm\n"
        
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

                # Determine replacement info for this slot
                replaced_name = failed_preassignments.get(position_name, {}).get(col_name, {}).get(i - 1)
                replaces_note = ""
                is_replacement = replaced_name is not None and name not in preassigned_names
                if is_replacement:
                    replaces_note = f"  ─ replaces {replaced_name}"

                # Pin mark: ⌖ = preassigned here, ⇆ = replaces a failed preassignment
                if name in preassigned_names:
                    pin_mark = "⌖ "
                elif is_replacement:
                    pin_mark = "⇆ "
                else:
                    pin_mark = "  "
                
                # Compute per-person penalty using position spec and reference height
                try:
                    candidate_row = castellers[castellers['Nom complet'] == name].iloc[0]
                    ref_h = reference_heights[col_name] if reference_heights and col_name in reference_heights else h
                    raw_score = _calculate_candidate_score(candidate_row, ref_h, position_spec, [castellers[castellers['Nom complet'] == s].iloc[0] for s in assigned if s and s != name], True)
                except Exception:
                    raw_score = 0.0

                if reference_heights:
                    ratio_pct = (h / reference_heights[col_name]) * 100
                    yield f"{pin_mark}{expertise_mark} {name:<16} {h:3.0f} cm   {ratio_pct:5.1f}%   penalty={raw_score:.2f}{replaces_note}\n"
                else:
                    yield f"{pin_mark}{name:<16} {h:3.0f} cm   penalty={raw_score:.2f}{replaces_note}\n"


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
    peripheral_stats: Optional[Dict] = None,
    preassigned_names: Optional[set] = None,
    preassigned_swapped_names: Optional[set] = None,
    failed_preassignments: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None,
) -> str:
    """Generate comprehensive summary of all assignments as a string."""
    if preassigned_names is None:
        preassigned_names = set()
    if preassigned_swapped_names is None:
        preassigned_swapped_names = set()
    if failed_preassignments is None:
        failed_preassignments = {}
    
    output = io.StringIO()
    
    # Print tronc position assignments first (structural order)
    for pos in ['baix', 'crossa', 'contrafort', 'agulla']:
        if pos in all_assignments:
            for line in _print_tronc_assignment(
                pos, all_assignments, columns, column_tronc_heights, 
                castellers, assignment_stats.get(pos) if assignment_stats else None,
                preassigned_names=preassigned_names,
                preassigned_swapped_names=preassigned_swapped_names,
                failed_preassignments=failed_preassignments,
            ):
                output.write(line)

    # Then print queue summaries
    for line in _print_queue_summary(
        all_assignments, castellers, columns, column_tronc_heights,
        peripheral_stats, preassigned_names=preassigned_names,
        preassigned_swapped_names=preassigned_swapped_names,
        failed_preassignments=failed_preassignments,
    ):
        output.write(line)
    
    # Validate
    errors, warnings = validate_structure(all_assignments, columns)
    
    if errors:
        output.write("="*60 + "\n")
        output.write("CRITICAL STRUCTURAL ERRORS:\n")
        for err in errors:
            output.write(f"✗ {err}\n")
        output.write("="*60 + "\n")
    
    if warnings:
        output.write("Warnings:\n")
        for warn in warnings:
            output.write(f"⚠ {warn}\n")
    
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
            output.write("\n##### UNASSIGNED CASTELLERS #####\n")
            for _, row in unassigned.iterrows():
                name = row.get('Nom complet', '')
                h = row.get('Alçada (cm)', None)
                pos1 = row.get('Posició 1', '')
                pos2 = row.get('Posició 2', '')
                expertise = ', '.join([p for p in [pos1, pos2] if p])
                height_str = f"{h:.1f} cm" if pd.notna(h) else "N/A"
                output.write(f"  - {name:<25} {height_str:8}  expertise={expertise}\n")
    except Exception as e:
        output.write(f"Failed to compute unassigned castellers list: {e}\n")
    
    return output.getvalue()


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