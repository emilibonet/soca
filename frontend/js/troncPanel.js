/**
 * Tronc & Pom panel — sidebar for trunk and top positions.
 */

import { state } from './state.js';
import { handleDragStart, handleDragEnd } from './dragDrop.js';
import { bus, Events } from './events.js';
import { showSmartSuggestions } from './algorithms.js';

// ── Tronc-Specific Drag Handlers ──────────────────────────────

function handleTroncDragOver(e) {
    e.preventDefault();
    const slot = e.currentTarget;
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

function handleTroncDragLeave(e) {
    const slot = e.currentTarget;
    slot.classList.remove('dragging-over', 'drag-replace');
}

function handleTroncDrop(e) {
    e.preventDefault();
    const slot = e.currentTarget;
    slot.classList.remove('dragging-over', 'drag-replace');

    const name = state.draggedCasteller;
    if (!name) return;

    const targetPosId = slot.dataset.positionId;
    state.assignToSlot(targetPosId, name);

    bus.emit(Events.UI_REFRESH);
    state.saveHistory(`Moved ${name} to ${targetPosId}`);
}

// ── Panel Initialization ──────────────────────────────────────

export function initTroncPanel() {
    const pomSection = document.getElementById('pomSection');
    const troncGrid = document.getElementById('troncGrid');

    // --- POM section ---
    // Enxaneta (top)
    const enxRow = document.createElement('div');
    enxRow.className = 'pom-row';
    enxRow.appendChild(createTroncSlot('enxaneta-top', 'Enxaneta'));
    pomSection.appendChild(enxRow);

    // Acotxador
    const acotRow = document.createElement('div');
    acotRow.className = 'pom-row';
    acotRow.appendChild(createTroncSlot('acotxador-top', 'Acotxador'));
    pomSection.appendChild(acotRow);

    // Dosos (2 side by side)
    const dosRow = document.createElement('div');
    dosRow.className = 'pom-row';
    dosRow.appendChild(createTroncSlot('dosos-Rengla-0', 'Dosos R'));
    dosRow.appendChild(createTroncSlot('dosos-Plena-Buida-0', 'Dosos P-B'));
    pomSection.appendChild(dosRow);

    // --- TRONC grid ---
    // Header row: empty + B + R + P
    const emptyHeader = document.createElement('div');
    troncGrid.appendChild(emptyHeader);
    ['Buida', 'Rengla', 'Plena'].forEach(col => {
        const colLabel = document.createElement('div');
        colLabel.className = 'tronc-col-label';
        colLabel.textContent = col[0];
        troncGrid.appendChild(colLabel);
    });

    // Rows: Terç, Segon, Baix (top to bottom)
    const rows = [
        { label: 'Terç',  type: 'terc' },
        { label: 'Segon', type: 'segon' },
        { label: 'Baix',  type: 'baix' }
    ];

    rows.forEach(row => {
        const rowLabel = document.createElement('div');
        rowLabel.className = 'tronc-row-label';
        rowLabel.textContent = row.label;
        troncGrid.appendChild(rowLabel);

        ['Buida', 'Rengla', 'Plena'].forEach(col => {
            const posId = `${row.type}-${col}`;
            troncGrid.appendChild(createTroncSlot(posId, `${row.label} ${col[0]}`));
        });
    });
}

// ── Tronc Slot Factory ────────────────────────────────────────

function createTroncSlot(positionId, labelText) {
    const slot = document.createElement('div');
    slot.className = 'tronc-slot';
    slot.dataset.positionId = positionId;
    slot.dataset.readOnly = 'false';

    const label = document.createElement('div');
    label.className = 'tronc-slot-label';
    label.textContent = labelText;
    slot.appendChild(label);

    // Drag & drop
    slot.addEventListener('dragover', handleTroncDragOver);
    slot.addEventListener('drop', handleTroncDrop);
    slot.addEventListener('dragleave', handleTroncDragLeave);
    slot.addEventListener('click', () => showSmartSuggestions(positionId));

    return slot;
}

// ── Panel Update ──────────────────────────────────────────────

export function updateTroncPanel() {
    document.querySelectorAll('.tronc-slot').forEach(slot => {
        const posId = slot.dataset.positionId;
        const name = state.getSlotContent(posId);
        const casteller = name ? state.getCasteller(name) : null;

        // Keep only the label
        const label = slot.querySelector('.tronc-slot-label');
        slot.innerHTML = '';
        if (label) slot.appendChild(label);

        if (casteller) {
            slot.classList.add('filled');

            const nameDiv = document.createElement('div');
            nameDiv.className = 'tronc-slot-name';
            nameDiv.textContent = casteller.name;
            slot.appendChild(nameDiv);

            const heightDiv = document.createElement('div');
            heightDiv.className = 'tronc-slot-height';
            heightDiv.textContent = `${casteller.height} cm`;
            slot.appendChild(heightDiv);

            // Make draggable
            slot.draggable = true;
            slot.ondragstart = handleDragStart;
            slot.ondragend = handleDragEnd;
        } else {
            slot.classList.remove('filled');
            slot.draggable = false;
            slot.ondragstart = null;
            slot.ondragend = null;
        }
    });
}
