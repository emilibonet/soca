/**
 * Dynamic queue creation and rebuild for mans and matrix (laterals + daus).
 */

import { state } from './state.js';
import { MANS_GEOMETRY, MATRIX_GEOMETRY, MATRIX_CONFIGS } from './config.js';
import { handleDragOver, handleDrop, handleDragLeave, handleDragStart, handleDragEnd } from './dragDrop.js';
import { updateMansSlot, updateMatrixSlot } from './slotUpdater.js';
import { showSmartSuggestions } from './algorithms.js';

// ── Mans Queue ────────────────────────────────────────────────

export function rebuildMansQueues() {
    const container = document.getElementById('canvasInner');

    // Remove all existing mans wrappers and old insert line
    container.querySelectorAll('.mans-wrapper').forEach(el => el.remove());
    const oldLine = container.querySelector('.mans-insert-line');
    if (oldLine) oldLine.remove();

    ['Rengla', 'Plena', 'Buida'].forEach(col => {
        const existing = state.assignments.mans?.[col] || [];
        // One wrapper per existing depth + one empty drop-target
        const totalSlots = existing.length + 1;

        for (let i = 0; i < totalSlots; i++) {
            const wrapper = createMansSlotSVG(col, i);
            container.appendChild(wrapper);
            updateMansSlot(`mans-${col}-${i}`);
        }
    });

    // Create reusable insert line indicator element
    const insertLine = document.createElement('div');
    insertLine.className = 'mans-insert-line';
    insertLine.id = 'mansInsertLine';
    container.appendChild(insertLine);
}

function createMansSlotSVG(col, depth) {
    const geo = MANS_GEOMETRY[col];
    const posId = `mans-${col}-${depth}`;
    const colClass = col === 'Rengla' ? 'col-rengla' : col === 'Plena' ? 'col-plena' : 'col-buida';

    const wrapper = document.createElement('div');
    wrapper.className = `mans-wrapper ${colClass}`;
    wrapper.dataset.positionId = posId;
    wrapper.dataset.positionType = 'mans';
    wrapper.dataset.mansCol = col;
    wrapper.dataset.mansDepth = depth;
    wrapper.dataset.readOnly = 'false';

    // Position based on depth
    const left = geo.baseLeft + depth * geo.dLeft;
    const top  = geo.baseTop  + depth * geo.dTop;
    wrapper.style.left   = `calc(50% + ${left}px)`;
    wrapper.style.top    = `calc(45% + ${top}px)`;
    wrapper.style.width  = `${geo.w}px`;
    wrapper.style.height = `${geo.h}px`;

    // Shallower depth = higher z-index (on top)
    wrapper.style.zIndex = 20 - depth;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'mans-rect');
    svg.setAttribute('viewBox', geo.viewBox);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');

    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', geo.points);
    poly.setAttribute('fill', 'var(--bg-card)');
    poly.setAttribute('stroke', geo.stroke);
    poly.setAttribute('stroke-width', '2');
    svg.appendChild(poly);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', geo.textX);
    label.setAttribute('y', geo.textY);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('class', 'slot-label');
    label.textContent = `Mans ${col[0]} ${depth + 1}`;
    svg.appendChild(label);

    const content = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    content.setAttribute('class', 'slot-content');
    svg.appendChild(content);

    wrapper.appendChild(svg);

    // Drag action label (hidden by default, shown during dragover)
    const dragLabel = document.createElement('div');
    dragLabel.className = 'mans-drag-label';
    wrapper.appendChild(dragLabel);

    wrapper.addEventListener('dragover', handleDragOver);
    wrapper.addEventListener('drop', handleDrop);
    wrapper.addEventListener('dragleave', handleDragLeave);
    wrapper.addEventListener('click', () => showSmartSuggestions(posId));

    return wrapper;
}

// ── Matrix Queues ─────────────────────────────────────────────

export function rebuildMatrixQueues() {
    const container = document.getElementById('canvasInner');

    // Remove old matrix slot wrappers, legacy containers, and old insert line
    container.querySelectorAll('.matrix-slot-wrapper').forEach(el => el.remove());
    container.querySelectorAll('.matrix-container').forEach(el => el.remove());
    const oldMatrixLine = container.querySelector('#matrixInsertLine');
    if (oldMatrixLine) oldMatrixLine.remove();

    MATRIX_CONFIGS.forEach(cfg => {
        const geo = MATRIX_GEOMETRY[cfg.id];
        if (!geo) return;

        // Determine max depth across all 3 columns
        const depths = cfg.columns.map(col => {
            const existing = state.assignments[col.queueType]?.[col.queueId] || [];
            return existing.length;
        });
        const maxDepth = Math.max(...depths, 0) + 1; // +1 for empty drop target row

        // Create SVG wrapper for each cell
        for (let d = 0; d < maxDepth; d++) {
            cfg.columns.forEach((col, colIdx) => {
                const wrapper = createMatrixSlotSVG(cfg, geo, col, colIdx, d);
                container.appendChild(wrapper);
                updateMatrixSlot(`${col.queueType}-${col.queueId}-${d}`);
            });
        }
    });

    // Create reusable matrix insert line indicator element
    const matrixLine = document.createElement('div');
    matrixLine.className = 'mans-insert-line';
    matrixLine.id = 'matrixInsertLine';
    container.appendChild(matrixLine);
}

function createMatrixSlotSVG(matrixCfg, geo, col, colIdx, depth) {
    const posId = `${col.queueType}-${col.queueId}-${depth}`;

    const wrapper = document.createElement('div');
    wrapper.className = 'matrix-slot-wrapper';
    wrapper.dataset.positionId = posId;
    wrapper.dataset.positionType = col.queueType;
    wrapper.dataset.matrixId = matrixCfg.id;
    wrapper.dataset.matrixQueueType = col.queueType;
    wrapper.dataset.matrixQueueId = col.queueId;
    wrapper.dataset.matrixDepth = String(depth);
    wrapper.dataset.matrixColIdx = String(colIdx);
    wrapper.dataset.readOnly = 'false';

    // Compute cell center position (offset from 50%/45%)
    const colOffset = colIdx - 1; // -1, 0, +1 for the three columns
    const cx = geo.originX + colOffset * geo.colSpacing * geo.edgeX + depth * geo.depthSpacing * geo.outX;
    const cy = geo.originY + colOffset * geo.colSpacing * geo.edgeY + depth * geo.depthSpacing * geo.outY;

    // Position wrapper's top-left corner
    const left = cx - geo.w / 2;
    const top = cy - geo.h / 2;
    wrapper.style.left = `calc(50% + ${left}px)`;
    wrapper.style.top = `calc(45% + ${top}px)`;
    wrapper.style.width = `${geo.w}px`;
    wrapper.style.height = `${geo.h}px`;

    // Shallower depth = higher z-index
    wrapper.style.zIndex = 20 - depth;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'matrix-slot-rect');
    svg.setAttribute('viewBox', geo.viewBox);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');

    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', geo.points);
    poly.setAttribute('fill', 'var(--bg-card)');
    poly.setAttribute('stroke', geo.strokeColors[colIdx]);
    poly.setAttribute('stroke-width', '2');
    svg.appendChild(poly);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', geo.textX);
    label.setAttribute('y', geo.textY);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('class', 'slot-label');
    label.textContent = `${col.header} ${depth + 1}`;
    svg.appendChild(label);

    const content = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    content.setAttribute('class', 'slot-content');
    svg.appendChild(content);

    wrapper.appendChild(svg);

    // Drag action label (hidden by default, shown during dragover)
    const dragLabel = document.createElement('div');
    dragLabel.className = 'matrix-drag-label';
    wrapper.appendChild(dragLabel);

    wrapper.addEventListener('dragover', handleDragOver);
    wrapper.addEventListener('drop', handleDrop);
    wrapper.addEventListener('dragleave', handleDragLeave);
    wrapper.addEventListener('click', () => showSmartSuggestions(posId));

    return wrapper;
}
