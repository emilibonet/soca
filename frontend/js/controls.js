/**
 * Controls — zoom/pan, undo/redo, import/export, and casteller CRUD modal.
 */

import { state } from './state.js';
import { bus, Events } from './events.js';
import { showToast } from './toast.js';
import { initPinya } from './pinya.js';

// Forward reference for updateAllSlots (set via init)
let _updateAllSlots = null;

export function setControlsUpdateFn(fn) {
    _updateAllSlots = fn;
}

// ── Zoom Controls ─────────────────────────────────────────────

export function zoomIn() {
    state.zoom = Math.min(state.zoom + 0.1, 3);
    updateZoom();
}

export function zoomOut() {
    state.zoom = Math.max(state.zoom - 0.1, 0.3);
    updateZoom();
}

export function zoomReset() {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    updateZoom();
}

export function updateZoom() {
    const inner = document.getElementById('canvasInner');
    inner.style.transform = `translate(calc(-50% + ${state.panX}px), calc(-50% + ${state.panY}px)) scale(${state.zoom})`;
}

// ── Pan & Scroll Zoom ─────────────────────────────────────────

export function initCanvasPanZoom() {
    const canvas = document.getElementById('canvas');
    let isPanning = false;
    let startX = 0, startY = 0;
    let startPanX = 0, startPanY = 0;

    canvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        if (e.target.closest('.slot, .casteller-card, .agulla-wrapper, .baix-wrapper, .contrafort-wrapper, .crossa-wrapper, .mans-wrapper, .matrix-slot-wrapper, .zoom-controls')) {
            if (!e.target.closest('.slot.tronc')) return;
        }
        isPanning = true;
        startX = e.clientX;
        startY = e.clientY;
        startPanX = state.panX;
        startPanY = state.panY;
        canvas.classList.add('grabbing');
        const inner = document.getElementById('canvasInner');
        inner.classList.add('panning');
        e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        state.panX = startPanX + dx;
        state.panY = startPanY + dy;
        updateZoom();
    });

    window.addEventListener('mouseup', () => {
        if (!isPanning) return;
        isPanning = false;
        canvas.classList.remove('grabbing');
        const inner = document.getElementById('canvasInner');
        inner.classList.remove('panning');
    });

    // Scroll-wheel zoom (centered on cursor)
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -0.08 : 0.08;
        const oldZoom = state.zoom;
        const newZoom = Math.min(Math.max(oldZoom + delta, 0.3), 3);
        if (newZoom === oldZoom) return;

        const rect = canvas.getBoundingClientRect();
        const cursorX = e.clientX - rect.left - rect.width / 2;
        const cursorY = e.clientY - rect.top - rect.height / 2;

        const scale = newZoom / oldZoom;
        state.panX = cursorX + scale * (state.panX - cursorX);
        state.panY = cursorY + scale * (state.panY - cursorY);
        state.zoom = newZoom;
        updateZoom();
    }, { passive: false });
}

// ── History ───────────────────────────────────────────────────

export function undo() {
    if (state.undo()) {
        bus.emit(Events.UI_REFRESH);
        showToast('Undo');
    }
}

export function redo() {
    if (state.redo()) {
        bus.emit(Events.UI_REFRESH);
        showToast('Redo');
    }
}

export function updateHistoryButtons() {
    const undoBtn = document.getElementById('undoBtn');
    const redoBtn = document.getElementById('redoBtn');
    if (undoBtn) undoBtn.disabled = state.historyIndex <= 0;
    if (redoBtn) redoBtn.disabled = state.historyIndex >= state.history.length - 1;
}

// ── Import / Export ───────────────────────────────────────────

export function exportAssignment() {
    const data = state.exportJSON();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const date = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
    a.download = `pinya_assignment_${date}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

export function importAssignment() {
    const input = document.getElementById('fileInput');
    input.click();
}

export function handleFileImport(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const data = JSON.parse(event.target.result);
            state.importJSON(data);
            if (_updateAllSlots) {
                initPinya(_updateAllSlots);
            }
            bus.emit(Events.UI_REFRESH);
            showToast('Assignment loaded successfully');
        } catch (error) {
            alert('Error loading file: ' + error.message);
        }
    };
    reader.readAsText(file);
    e.target.value = '';
}

// ── Casteller CRUD Modal ──────────────────────────────────────

let editingCasteller = null;

export function openCastellerModal(casteller = null) {
    editingCasteller = casteller;
    const modal = document.getElementById('castellerModal');
    const title = document.getElementById('modalTitle');
    const form = document.getElementById('castellerForm');

    title.textContent = casteller ? 'Edit Casteller' : 'Add Casteller';

    if (casteller) {
        document.getElementById('formName').value = casteller.name;
        document.getElementById('formHeight').value = casteller.height;
        document.getElementById('formPosition1').value = casteller.position_1 || '';
        document.getElementById('formPosition2').value = casteller.position_2 || '';
        document.getElementById('formWeight').value = casteller.weight || '';
    } else {
        form.reset();
    }

    modal.classList.add('active');
}

export function closeModal() {
    document.getElementById('castellerModal').classList.remove('active');
    editingCasteller = null;
}

export function saveCasteller(e) {
    e.preventDefault();

    const newCasteller = {
        name: document.getElementById('formName').value.trim(),
        height: parseFloat(document.getElementById('formHeight').value),
        position_1: document.getElementById('formPosition1').value,
        position_2: document.getElementById('formPosition2').value,
        weight: document.getElementById('formWeight').value ? parseFloat(document.getElementById('formWeight').value) : null
    };

    try {
        if (editingCasteller) {
            state.updateCasteller(editingCasteller.name, newCasteller);
        } else {
            state.addCasteller(newCasteller);
        }
        closeModal();
        bus.emit(Events.UI_REFRESH);
    } catch (error) {
        alert(error.message);
    }
}

export function editCasteller(name) {
    const casteller = state.getCasteller(name);
    if (casteller) {
        openCastellerModal(casteller);
    }
}

export function removeCasteller(name) {
    if (confirm(`Remove ${name} from the cast? This cannot be undone.`)) {
        state.removeCasteller(name);
        bus.emit(Events.UI_REFRESH);
    }
}
