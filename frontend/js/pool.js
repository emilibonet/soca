/**
 * Pool panel — lists unassigned castellers with search, sort, and drag support.
 */

import { state } from './state.js';
import { handleDragStart, handleDragEnd, hideInsertLine } from './dragDrop.js';
import { showToast } from './toast.js';

// ── Forward references (set via setPoolCallbacks) ─────────────
let _editCasteller = null;
let _removeCasteller = null;

/**
 * Inject callbacks to avoid circular dependency with controls.js.
 * Called once during initialization from main.js.
 */
export function setPoolCallbacks({ editCasteller, removeCasteller }) {
    _editCasteller = editCasteller;
    _removeCasteller = removeCasteller;
}

// ── Pool Rendering ────────────────────────────────────────────

export function updatePool() {
    const poolList = document.getElementById('poolList');
    const search = document.getElementById('poolSearch').value.toLowerCase();
    const sortBy = document.getElementById('poolSort').value;

    let unassigned = state.getUnassigned();

    // Filter
    if (search) {
        unassigned = unassigned.filter(c =>
            c.name.toLowerCase().includes(search) ||
            (c.position_1 || '').toLowerCase().includes(search) ||
            (c.position_2 || '').toLowerCase().includes(search)
        );
    }

    // Sort
    unassigned.sort((a, b) => {
        if (sortBy === 'name') return a.name.localeCompare(b.name);
        if (sortBy === 'height') return b.height - a.height;
        if (sortBy === 'position') return (a.position_1 || '').localeCompare(b.position_1 || '');
        return 0;
    });

    poolList.innerHTML = '';
    unassigned.forEach(casteller => {
        const card = createCastellerCard(casteller);
        poolList.appendChild(card);
    });

    // Update count
    const poolCount = document.getElementById('poolCount');
    poolCount.textContent = `${unassigned.length} / ${state.castellers.length}`;
}

// ── Casteller Card ────────────────────────────────────────────

function createCastellerCard(casteller) {
    const card = document.createElement('div');
    card.className = 'casteller-card';
    card.draggable = true;
    card.dataset.name = casteller.name;

    card.addEventListener('dragstart', (e) => {
        state.draggedCasteller = casteller.name;
        state.draggedElement = card;
        state.dragSourcePosId = null; // Pool cards have no source position
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', casteller.name);
    });

    card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
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
    });

    const name = document.createElement('div');
    name.className = 'casteller-name';
    name.textContent = casteller.name;
    card.appendChild(name);

    const info = document.createElement('div');
    info.className = 'casteller-info';
    info.innerHTML = `
        <span>${casteller.height} cm</span>
        <span>${casteller.position_1 || '\u2014'}</span>
    `;
    card.appendChild(info);

    const actions = document.createElement('div');
    actions.className = 'casteller-actions';

    const editBtn = document.createElement('button');
    editBtn.className = 'icon-btn';
    editBtn.textContent = '\u270E Edit';
    editBtn.onclick = () => _editCasteller && _editCasteller(casteller.name);
    actions.appendChild(editBtn);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'icon-btn';
    removeBtn.textContent = '\u00D7 Remove';
    removeBtn.onclick = () => _removeCasteller && _removeCasteller(casteller.name);
    actions.appendChild(removeBtn);

    card.appendChild(actions);

    return card;
}
