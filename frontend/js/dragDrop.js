/**
 * Drag-and-drop handlers for all slot types.
 * Also contains mans/matrix drop-action detection and insert-line helpers.
 */

import { state } from './state.js';
import { MANS_GEOMETRY, MATRIX_GEOMETRY } from './config.js';
import { bus, Events } from './events.js';
import { showToast } from './toast.js';

// ── Drag Start / End ──────────────────────────────────────────

export function handleDragStart(e) {
    const slot = e.currentTarget;
    const posId = slot.dataset.positionId;
    const name = state.getSlotContent(posId);

    if (!name) return;

    state.draggedCasteller = name;
    state.draggedElement = slot;
    state.dragSourcePosId = posId;
    slot.style.opacity = '0.5';
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', name);
}

export function handleDragEnd(e) {
    e.currentTarget.style.opacity = '1';
    state.draggedCasteller = null;
    state.draggedElement = null;
    state.dragSourcePosId = null;
    // Clean up any lingering mans drag feedback
    hideInsertLine();
    document.querySelectorAll('.mans-wrapper.drag-substitute, .mans-wrapper.drag-insert').forEach(w => {
        w.classList.remove('drag-substitute', 'drag-insert');
        const lbl = w.querySelector('.mans-drag-label');
        if (lbl) lbl.textContent = '';
    });
    // Clean up matrix drag feedback
    document.querySelectorAll('.matrix-slot-wrapper.drag-substitute, .matrix-slot-wrapper.drag-insert').forEach(w => {
        w.classList.remove('drag-substitute', 'drag-insert');
        const lbl = w.querySelector('.matrix-drag-label');
        if (lbl) lbl.textContent = '';
    });
}

// ── Drag Over ─────────────────────────────────────────────────

export function handleDragOver(e) {
    e.preventDefault();
    const slot = e.currentTarget;

    if (slot.dataset.readOnly === 'true') {
        slot.classList.add('drag-invalid');
        return;
    }

    // Mans queue: detect substitute vs insert based on cursor position
    if (slot.classList.contains('mans-wrapper')) {
        slot.classList.remove('dragging-over', 'drag-replace', 'drag-substitute', 'drag-insert');
        const { action, insertIndex } = getMansDropAction(slot, e);
        slot.dataset.dropAction = action;
        slot.dataset.insertIndex = insertIndex !== null ? insertIndex : '';

        if (action === 'substitute') {
            slot.classList.add('drag-substitute');
            const label = slot.querySelector('.mans-drag-label');
            if (label) label.textContent = '\u21C4 Replace';
            hideInsertLine();
        } else {
            slot.classList.add('drag-insert');
            const label = slot.querySelector('.mans-drag-label');
            if (label) label.textContent = '\u2935 Insert';
            showInsertLine(slot.dataset.mansCol, insertIndex);
        }
        e.dataTransfer.dropEffect = 'move';
        return;
    }

    // Matrix slot: detect substitute vs insert based on cursor position
    if (slot.classList.contains('matrix-slot-wrapper')) {
        slot.classList.remove('dragging-over', 'drag-replace', 'drag-substitute', 'drag-insert');
        const { action, insertIndex } = getMatrixDropAction(slot, e);
        slot.dataset.dropAction = action;
        slot.dataset.insertIndex = insertIndex !== null ? insertIndex : '';

        if (action === 'substitute') {
            slot.classList.add('drag-substitute');
            const label = slot.querySelector('.matrix-drag-label');
            if (label) label.textContent = '\u21C4 Replace';
            hideMatrixInsertLine();
        } else {
            slot.classList.add('drag-insert');
            const label = slot.querySelector('.matrix-drag-label');
            if (label) label.textContent = '\u2935 Insert';
            showMatrixInsertLine(slot.dataset.matrixId, slot.dataset.matrixQueueId, slot.dataset.matrixQueueType, parseInt(slot.dataset.matrixColIdx), insertIndex);
        }
        e.dataTransfer.dropEffect = 'move';
        return;
    }

    // Non-queue: detect if slot is filled → show replace feedback
    const posId = slot.dataset.positionId;
    const isFilled = posId && state.getSlotContent(posId);
    slot.classList.remove('dragging-over', 'drag-replace');
    if (isFilled) {
        slot.classList.add('drag-replace');
    } else {
        slot.classList.add('dragging-over');
    }
    e.dataTransfer.dropEffect = 'move';
}

// ── Drag Leave ────────────────────────────────────────────────

export function handleDragLeave(e) {
    const slot = e.currentTarget;
    slot.classList.remove('dragging-over', 'drag-invalid', 'drag-replace', 'drag-substitute', 'drag-insert', 'drag-insert-before', 'drag-insert-after');
    if (slot.classList.contains('mans-wrapper')) {
        delete slot.dataset.dropAction;
        delete slot.dataset.insertIndex;
        const lbl = slot.querySelector('.mans-drag-label');
        if (lbl) lbl.textContent = '';
        hideInsertLine();
    }
    if (slot.classList.contains('matrix-slot-wrapper')) {
        delete slot.dataset.dropAction;
        delete slot.dataset.insertIndex;
        const lbl = slot.querySelector('.matrix-drag-label');
        if (lbl) lbl.textContent = '';
        hideMatrixInsertLine();
    }
}

// ── Drop ──────────────────────────────────────────────────────

export function handleDrop(e) {
    e.preventDefault();
    const slot = e.currentTarget;
    slot.classList.remove('dragging-over', 'drag-invalid', 'drag-replace', 'drag-substitute', 'drag-insert', 'drag-insert-before', 'drag-insert-after');

    if (slot.dataset.readOnly === 'true') {
        showToast('Tronc positions are read-only');
        return;
    }

    const name = state.draggedCasteller;
    if (!name) return;

    const targetPosId = slot.dataset.positionId;
    const [posType, col, idx] = state.parsePositionId(targetPosId);

    // Queue-style drop (mans + matrix daus/laterals): substitute or insert
    const isQueueDrop = slot.classList.contains('mans-wrapper') || slot.classList.contains('matrix-slot-wrapper');
    if (isQueueDrop && idx !== undefined) {
        const dropAction = slot.dataset.dropAction || 'substitute';
        let insertIndex = (slot.dataset.insertIndex !== undefined && slot.dataset.insertIndex !== '')
            ? parseInt(slot.dataset.insertIndex) : null;

        // Clean up drag state
        delete slot.dataset.dropAction;
        delete slot.dataset.insertIndex;
        if (slot.classList.contains('mans-wrapper')) {
            const lbl = slot.querySelector('.mans-drag-label');
            if (lbl) lbl.textContent = '';
            hideInsertLine();
        }
        if (slot.classList.contains('matrix-slot-wrapper')) {
            const lbl = slot.querySelector('.matrix-drag-label');
            if (lbl) lbl.textContent = '';
            hideMatrixInsertLine();
        }

        // No-op: dragging onto own slot with substitute
        if (dropAction === 'substitute' && state.dragSourcePosId === targetPosId) {
            bus.emit(Events.UI_REFRESH);
            return;
        }

        // Save source info before removal (for index adjustment)
        const sourceInfo = state.dragSourcePosId
            ? state.parsePositionId(state.dragSourcePosId) : null;

        // Remove from previous position
        state.removeFromAssignments(name);

        // Ensure the queue array exists
        if (!state.assignments[posType][col]) {
            state.assignments[posType][col] = [];
        }

        if (dropAction === 'insert' && insertIndex !== null) {
            // Adjust insert index if source was in the same queue column
            if (sourceInfo && sourceInfo[0] === posType && sourceInfo[1] === col
                && sourceInfo[2] !== undefined && sourceInfo[2] < insertIndex) {
                insertIndex--;
            }
            while (state.assignments[posType][col].length < insertIndex) {
                state.assignments[posType][col].push([null]);
            }
            state.assignments[posType][col].splice(insertIndex, 0, [name]);
        } else {
            let adjustedIdx = idx;
            if (sourceInfo && sourceInfo[0] === posType && sourceInfo[1] === col
                && sourceInfo[2] !== undefined && sourceInfo[2] < idx) {
                adjustedIdx--;
            }
            while (state.assignments[posType][col].length <= adjustedIdx) {
                state.assignments[posType][col].push([null]);
            }
            state.assignments[posType][col][adjustedIdx] = [name];
        }

        // Clean up trailing nulls
        while (state.assignments[posType][col].length > 0
            && (!state.assignments[posType][col][state.assignments[posType][col].length - 1]
                || state.assignments[posType][col][state.assignments[posType][col].length - 1][0] === null)) {
            state.assignments[posType][col].pop();
        }
    } else {
        // Default behavior for non-queue positions
        state.assignToSlot(targetPosId, name);
    }

    bus.emit(Events.UI_REFRESH);
    state.saveHistory(`Moved ${name} to ${targetPosId}`);
}

// ── Mans Drop-Action Detection ────────────────────────────────

export function getMansDropAction(wrapper, e) {
    const col = wrapper.dataset.mansCol;
    const depth = parseInt(wrapper.dataset.mansDepth);
    const geo = MANS_GEOMETRY[col];
    const posId = wrapper.dataset.positionId;
    const existingName = state.getSlotContent(posId);

    // Empty slot → always fill (substitute)
    if (!existingName) return { action: 'substitute', insertIndex: null };

    // Get cursor position relative to wrapper center
    const rect = wrapper.getBoundingClientRect();
    const cx = e.clientX - (rect.left + rect.width / 2);
    const cy = e.clientY - (rect.top + rect.height / 2);

    // Queue direction vector
    const dx = geo.dLeft, dy = geo.dTop;
    const len = Math.sqrt(dx * dx + dy * dy);

    // Project cursor offset onto queue direction (signed distance from center)
    const proj = (cx * dx + cy * dy) / len;

    // Compute half-extent of the slot polygon along queue direction
    const points = geo.points.split(/\s+/).map(p => {
        const [x, y] = p.split(',').map(Number);
        return { x: x - geo.w / 2, y: y - geo.h / 2 };
    });
    const projections = points.map(p => (p.x * dx + p.y * dy) / len);
    const halfExtent = Math.max(...projections.map(Math.abs));

    // Normalize: -1 = leading edge (toward center), +1 = trailing edge (outward)
    const normalized = proj / halfExtent;

    // Thresholds: center 50% = substitute, edges 25% each = insert
    if (normalized < -0.4) return { action: 'insert', insertIndex: depth };
    if (normalized > 0.4)  return { action: 'insert', insertIndex: depth + 1 };
    return { action: 'substitute', insertIndex: null };
}

// ── Matrix Drop-Action Detection ──────────────────────────────

export function getMatrixDropAction(wrapper, e) {
    const depth = parseInt(wrapper.dataset.matrixDepth);
    const posId = wrapper.dataset.positionId;
    const existingName = state.getSlotContent(posId);

    // Empty slot → always substitute (fill)
    if (!existingName) return { action: 'substitute', insertIndex: null };

    const matrixId = wrapper.dataset.matrixId;
    const geo = MATRIX_GEOMETRY[matrixId];

    // Get cursor position relative to wrapper center
    const rect = wrapper.getBoundingClientRect();
    const cx = e.clientX - (rect.left + rect.width / 2);
    const cy = e.clientY - (rect.top + rect.height / 2);

    // Project cursor onto outward direction
    const proj = cx * geo.outX + cy * geo.outY;
    const halfDepth = geo.depthSpacing / 2;
    const normalized = proj / halfDepth;

    if (normalized < -0.4) return { action: 'insert', insertIndex: depth };
    if (normalized > 0.4)  return { action: 'insert', insertIndex: depth + 1 };
    return { action: 'substitute', insertIndex: null };
}

// ── Insert Line Helpers ───────────────────────────────────────

export function showInsertLine(col, insertIndex) {
    const line = document.getElementById('mansInsertLine');
    if (!line) return;

    const geo = MANS_GEOMETRY[col];

    if (col === 'Rengla') {
        const left = geo.baseLeft - 10;
        const top  = geo.baseTop + insertIndex * geo.dTop;
        line.style.left      = `calc(50% + ${left}px)`;
        line.style.top       = `calc(45% + ${top - 2}px)`;
        line.style.width     = `${geo.w + 20}px`;
        line.style.height    = '4px';
        line.style.transform = 'none';
    } else if (col === 'Buida') {
        const left = geo.baseLeft + (insertIndex - 1) * geo.dLeft + 30;
        const top  = geo.baseTop  + (insertIndex - 1) * geo.dTop  + 52;
        const edgeAngle = Math.atan2(104, -60) * 180 / Math.PI;
        line.style.left      = `calc(50% + ${left}px)`;
        line.style.top       = `calc(45% + ${top}px)`;
        line.style.width     = '130px';
        line.style.height    = '4px';
        line.style.transform = `translate(-50%, -50%) rotate(${edgeAngle}deg)`;
    } else { // Plena
        const left = geo.baseLeft + (insertIndex - 1) * geo.dLeft + 78;
        const top  = geo.baseTop  + (insertIndex - 1) * geo.dTop  + 52;
        const edgeAngle = Math.atan2(104, 60) * 180 / Math.PI;
        line.style.left      = `calc(50% + ${left}px)`;
        line.style.top       = `calc(45% + ${top}px)`;
        line.style.width     = '130px';
        line.style.height    = '4px';
        line.style.transform = `translate(-50%, -50%) rotate(${edgeAngle}deg)`;
    }

    line.classList.add('visible');
}

export function hideInsertLine() {
    const line = document.getElementById('mansInsertLine');
    if (line) line.classList.remove('visible');
}

export function showMatrixInsertLine(matrixId, queueId, queueType, colIdx, insertIndex) {
    const line = document.getElementById('matrixInsertLine');
    if (!line) return;

    const geo = MATRIX_GEOMETRY[matrixId];
    if (!geo) return;

    const colOffset = colIdx - 1;

    const cx = geo.originX + colOffset * geo.colSpacing * geo.edgeX + (insertIndex - 0.5) * geo.depthSpacing * geo.outX;
    const cy = geo.originY + colOffset * geo.colSpacing * geo.edgeY + (insertIndex - 0.5) * geo.depthSpacing * geo.outY;

    const edgeAngle = Math.atan2(geo.edgeY, geo.edgeX) * 180 / Math.PI;
    const longSideLen = 130;

    line.style.left      = `calc(50% + ${cx}px)`;
    line.style.top       = `calc(45% + ${cy}px)`;
    line.style.width     = `${longSideLen}px`;
    line.style.height    = '4px';
    line.style.transform = `translate(-50%, -50%) rotate(${edgeAngle}deg)`;

    line.classList.add('visible');
}

export function hideMatrixInsertLine() {
    const line = document.getElementById('matrixInsertLine');
    if (line) line.classList.remove('visible');
}
