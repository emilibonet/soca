/**
 * Slot content update functions and height/expertise indicators.
 * Each function populates a slot's SVG or HTML content based on current state.
 */

import { state } from './state.js';
import { handleDragStart, handleDragEnd } from './dragDrop.js';

// ── Position Keywords (for expertise matching) ────────────────

const POSITION_KEYWORDS = {
    baix: ['Baix'],
    crossa: ['Crossa'],
    contrafort: ['Contrafort'],
    agulla: ['Agulla'],
    mans: ['Primeres'],
    daus: ['Dau/Vent', 'Dau', 'Vent'],
    laterals: ['Lateral']
};

export function getPositionKeywords(posType) {
    return POSITION_KEYWORDS[posType] || [];
}

// ── Height Ratio Calculations ─────────────────────────────────

export function calculateHeightRatio(positionId, casteller) {
    const refHeight = getReferenceHeight(positionId);
    if (refHeight === null || refHeight === 0) return null;
    return casteller.height / refHeight;
}

export function getReferenceHeight(positionId) {
    const [posType, col, idx] = state.parsePositionId(positionId);

    if (posType === 'crossa' || posType === 'contrafort') {
        const baixName = state.getSlotContent(`baix-${col}`);
        const baix = baixName ? state.getCasteller(baixName) : null;
        return baix ? baix.height : null;
    }

    if (posType === 'agulla') {
        const baixName = state.getSlotContent(`baix-${col}`);
        const segonName = state.getSlotContent(`segon-${col}`);
        const baix = baixName ? state.getCasteller(baixName) : null;
        const segon = segonName ? state.getCasteller(segonName) : null;
        if (!baix || !segon) return null;
        return baix.height + segon.height;
    }

    if (posType === 'mans' || posType === 'laterals') {
        const baseCol = posType === 'laterals' ? col.replace(/-(?:left|right)$/, '') : col;
        if (idx === 0) {
            const baixName = state.getSlotContent(`baix-${baseCol}`);
            const segonName = state.getSlotContent(`segon-${baseCol}`);
            const baix = baixName ? state.getCasteller(baixName) : null;
            const segon = segonName ? state.getCasteller(segonName) : null;
            if (!baix || !segon) return null;
            return baix.height + segon.height;
        } else {
            const prevName = state.getSlotContent(`${posType}-${col}-${idx - 1}`);
            const prev = prevName ? state.getCasteller(prevName) : null;
            return prev ? prev.height : null;
        }
    }

    if (posType === 'daus') {
        if (idx === 0) {
            const cols = col.split('\u2194');
            const heights = [];
            for (const c of cols) {
                const normalCol = c === 'R' ? 'Rengla' : c === 'P' ? 'Plena' : 'Buida';
                const baixName = state.getSlotContent(`baix-${normalCol}`);
                const segonName = state.getSlotContent(`segon-${normalCol}`);
                const baix = baixName ? state.getCasteller(baixName) : null;
                const segon = segonName ? state.getCasteller(segonName) : null;
                if (baix && segon) heights.push(baix.height + segon.height);
            }
            if (heights.length === 0) return null;
            return heights.reduce((a, b) => a + b) / heights.length;
        } else {
            const prevName = state.getSlotContent(`${posType}-${col}-${idx - 1}`);
            const prev = prevName ? state.getCasteller(prevName) : null;
            return prev ? prev.height : null;
        }
    }

    return null;
}

export function getHeightRatioRange(posType, depth) {
    if (posType === 'crossa') {
        const c = state.config.positions.crossa;
        return { min: c.height_ratio_min, max: c.height_ratio_max };
    }
    if (posType === 'contrafort') {
        const c = state.config.positions.contrafort;
        return { min: c.height_ratio_min, max: c.height_ratio_max };
    }
    if (posType === 'agulla') {
        const c = state.config.positions.agulla;
        return { min: c.height_ratio_min, max: c.height_ratio_max };
    }
    if (['mans', 'daus', 'laterals'].includes(posType)) {
        const qConfig = state.config.queues[posType];
        if (depth === 0) {
            return { min: qConfig.height_ratio_min, max: qConfig.height_ratio_max };
        } else {
            return { min: qConfig.queue_height_ratio_min, max: qConfig.queue_height_ratio_max };
        }
    }
    return { min: null, max: null };
}

// ── Indicator Creation ────────────────────────────────────────

export function getHeightIndicator(positionId, casteller) {
    const [posType, col, idx] = state.parsePositionId(positionId);

    if (posType === 'baix') return null;

    const ratio = calculateHeightRatio(positionId, casteller);
    if (ratio === null) return null;

    const { min, max } = getHeightRatioRange(posType, idx);
    if (min === null || max === null) return null;

    const indicator = document.createElement('div');
    indicator.className = 'indicator';
    indicator.textContent = '\u25CF';
    indicator.title = `Height ratio: ${(ratio * 100).toFixed(1)}% (target: ${(min * 100).toFixed(0)}\u2013${(max * 100).toFixed(0)}%)`;

    if (ratio >= min && ratio <= max) {
        indicator.classList.add('height-good');
    } else if (ratio >= min * 0.95 && ratio <= max * 1.05) {
        indicator.classList.add('height-ok');
    } else {
        indicator.classList.add('height-bad');
    }

    return indicator;
}

export function getExpertiseIndicator(positionId, casteller) {
    const [posType] = state.parsePositionId(positionId);

    const keywords = getPositionKeywords(posType);
    const pos1 = (casteller.position_1 || '').toLowerCase();
    const pos2 = (casteller.position_2 || '').toLowerCase();

    const indicator = document.createElement('div');
    indicator.className = 'indicator';

    let hasPrimary = false;
    let hasSecondary = false;

    keywords.forEach(kw => {
        if (pos1.includes(kw.toLowerCase())) hasPrimary = true;
        if (pos2.includes(kw.toLowerCase())) hasSecondary = true;
    });

    if (hasPrimary) {
        indicator.classList.add('expertise-primary');
        indicator.textContent = '\u2B50';
        indicator.title = `Primary expertise: ${casteller.position_1}`;
    } else if (hasSecondary) {
        indicator.classList.add('expertise-secondary');
        indicator.textContent = '\u2606';
        indicator.title = `Secondary expertise: ${casteller.position_2}`;
    } else {
        indicator.classList.add('expertise-none');
        indicator.textContent = '\u25CB';
        indicator.title = 'No matching expertise';
    }

    return indicator;
}

// ── SVG Indicator Helper ──────────────────────────────────────

function appendSVGIndicators(content, tx, ty, positionId, casteller, showHeight) {
    if (showHeight) {
        const hInd = getHeightIndicator(positionId, casteller);
        const eInd = getExpertiseIndicator(positionId, casteller);
        let indX = tx - 10;
        [hInd, eInd].forEach(ind => {
            if (!ind) return;
            const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            t.setAttribute('x', indX);
            t.setAttribute('y', ty + 14);
            t.setAttribute('text-anchor', 'middle');
            t.setAttribute('class', 'indicator-text');
            t.textContent = ind.textContent;
            t.setAttribute('fill', window.getComputedStyle(ind).color || 'white');
            t.setAttribute('title', ind.title);
            content.appendChild(t);
            indX += 20;
        });
    } else {
        const eInd = getExpertiseIndicator(positionId, casteller);
        if (eInd) {
            const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            t.setAttribute('x', tx);
            t.setAttribute('y', ty + 14);
            t.setAttribute('text-anchor', 'middle');
            t.setAttribute('class', 'indicator-text');
            t.textContent = eInd.textContent;
            t.setAttribute('fill', window.getComputedStyle(eInd).color || 'white');
            t.setAttribute('title', eInd.title);
            content.appendChild(t);
        }
    }
}

// ── HTML Slot Update ──────────────────────────────────────────

export function updateSlot(positionId) {
    const slot = document.querySelector(`[data-position-id="${positionId}"]`);
    if (!slot) return;

    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;

    const label = slot.querySelector('.slot-label');
    slot.innerHTML = '';
    if (label) slot.appendChild(label);

    if (casteller) {
        slot.classList.add('filled');

        const nameDiv = document.createElement('div');
        nameDiv.className = 'slot-name';
        nameDiv.textContent = casteller.name;
        slot.appendChild(nameDiv);

        const heightDiv = document.createElement('div');
        heightDiv.className = 'slot-height';
        heightDiv.textContent = `${casteller.height} cm`;
        slot.appendChild(heightDiv);

        const indicators = document.createElement('div');
        indicators.className = 'slot-indicators';

        const heightIndicator = getHeightIndicator(positionId, casteller);
        if (heightIndicator) indicators.appendChild(heightIndicator);

        const expertiseIndicator = getExpertiseIndicator(positionId, casteller);
        indicators.appendChild(expertiseIndicator);
        slot.appendChild(indicators);

        if (!slot.dataset.readOnly || slot.dataset.readOnly === 'false') {
            slot.draggable = true;
            slot.addEventListener('dragstart', handleDragStart);
            slot.addEventListener('dragend', handleDragEnd);
        }
    } else {
        slot.classList.remove('filled');
        slot.draggable = false;
    }
}

// ── SVG Slot Updaters ─────────────────────────────────────────

export function updateAgullaSlot(positionId) {
    const wrapper = document.querySelector(`.agulla-wrapper[data-position-id="${positionId}"]`);
    if (!wrapper) return;

    const svg = wrapper.querySelector('.agulla-triangle');
    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;
    const content = svg.querySelector('.slot-content');
    const label = svg.querySelector('.slot-label');

    content.innerHTML = '';

    const tx = parseFloat(label.getAttribute('x'));
    const ty = parseFloat(label.getAttribute('y'));

    if (casteller) {
        wrapper.classList.add('filled');
        label.setAttribute('opacity', '0.3');

        const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        nameEl.setAttribute('x', tx);
        nameEl.setAttribute('y', ty - 12);
        nameEl.setAttribute('text-anchor', 'middle');
        nameEl.setAttribute('class', 'slot-name');
        nameEl.textContent = casteller.name;
        content.appendChild(nameEl);

        const heightEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        heightEl.setAttribute('x', tx);
        heightEl.setAttribute('y', ty + 2);
        heightEl.setAttribute('text-anchor', 'middle');
        heightEl.setAttribute('class', 'slot-height');
        heightEl.textContent = `${casteller.height} cm`;
        content.appendChild(heightEl);

        appendSVGIndicators(content, tx, ty, positionId, casteller, true);

        wrapper.draggable = true;
        wrapper.ondragstart = handleDragStart;
        wrapper.ondragend = handleDragEnd;
    } else {
        wrapper.classList.remove('filled');
        wrapper.draggable = false;
        wrapper.ondragstart = null;
        wrapper.ondragend = null;
        label.setAttribute('opacity', '1');
    }
}

export function updateBaixSlot(positionId) {
    const wrapper = document.querySelector(`.baix-wrapper[data-position-id="${positionId}"]`);
    if (!wrapper) return;

    const svg = wrapper.querySelector('.baix-rect');
    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;
    const content = svg.querySelector('.slot-content');
    const label = svg.querySelector('.slot-label');

    content.innerHTML = '';

    const tx = parseFloat(label.getAttribute('x'));
    const ty = parseFloat(label.getAttribute('y'));

    if (casteller) {
        wrapper.classList.add('filled');
        label.setAttribute('opacity', '0.3');

        const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        nameEl.setAttribute('x', tx);
        nameEl.setAttribute('y', ty - 12);
        nameEl.setAttribute('text-anchor', 'middle');
        nameEl.setAttribute('class', 'slot-name');
        nameEl.textContent = casteller.name;
        content.appendChild(nameEl);

        const heightEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        heightEl.setAttribute('x', tx);
        heightEl.setAttribute('y', ty + 2);
        heightEl.setAttribute('text-anchor', 'middle');
        heightEl.setAttribute('class', 'slot-height');
        heightEl.textContent = `${casteller.height} cm`;
        content.appendChild(heightEl);

        // Expertise only (no height ratio for baix)
        appendSVGIndicators(content, tx, ty, positionId, casteller, false);

        wrapper.draggable = true;
        wrapper.ondragstart = handleDragStart;
        wrapper.ondragend = handleDragEnd;
    } else {
        wrapper.classList.remove('filled');
        wrapper.draggable = false;
        wrapper.ondragstart = null;
        wrapper.ondragend = null;
        label.setAttribute('opacity', '1');
    }
}

export function updateMansSlot(positionId) {
    const wrapper = document.querySelector(`.mans-wrapper[data-position-id="${positionId}"]`);
    if (!wrapper) return;

    const svg = wrapper.querySelector('.mans-rect');
    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;
    const content = svg.querySelector('.slot-content');
    const label = svg.querySelector('.slot-label');

    content.innerHTML = '';

    const tx = parseFloat(label.getAttribute('x'));
    const ty = parseFloat(label.getAttribute('y'));

    if (casteller) {
        wrapper.classList.add('filled');
        label.setAttribute('opacity', '0.3');

        const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        nameEl.setAttribute('x', tx);
        nameEl.setAttribute('y', ty - 12);
        nameEl.setAttribute('text-anchor', 'middle');
        nameEl.setAttribute('class', 'slot-name');
        nameEl.textContent = casteller.name;
        content.appendChild(nameEl);

        const heightEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        heightEl.setAttribute('x', tx);
        heightEl.setAttribute('y', ty + 2);
        heightEl.setAttribute('text-anchor', 'middle');
        heightEl.setAttribute('class', 'slot-height');
        heightEl.textContent = `${casteller.height} cm`;
        content.appendChild(heightEl);

        appendSVGIndicators(content, tx, ty, positionId, casteller, true);

        wrapper.draggable = true;
        wrapper.ondragstart = handleDragStart;
        wrapper.ondragend = handleDragEnd;
    } else {
        wrapper.classList.remove('filled');
        wrapper.draggable = false;
        wrapper.ondragstart = null;
        wrapper.ondragend = null;
        label.setAttribute('opacity', '1');
    }
}

export function updateContrafortSlot(positionId) {
    const wrapper = document.querySelector(`.contrafort-wrapper[data-position-id="${positionId}"]`);
    if (!wrapper) return;

    const svg = wrapper.querySelector('.contrafort-rect');
    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;
    const content = svg.querySelector('.slot-content');
    const label = svg.querySelector('.slot-label');

    content.innerHTML = '';

    const tx = parseFloat(label.getAttribute('x'));
    const ty = parseFloat(label.getAttribute('y'));

    if (casteller) {
        wrapper.classList.add('filled');
        label.setAttribute('opacity', '0.3');

        const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        nameEl.setAttribute('x', tx);
        nameEl.setAttribute('y', ty - 12);
        nameEl.setAttribute('text-anchor', 'middle');
        nameEl.setAttribute('class', 'slot-name');
        nameEl.textContent = casteller.name;
        content.appendChild(nameEl);

        const heightEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        heightEl.setAttribute('x', tx);
        heightEl.setAttribute('y', ty + 2);
        heightEl.setAttribute('text-anchor', 'middle');
        heightEl.setAttribute('class', 'slot-height');
        heightEl.textContent = `${casteller.height} cm`;
        content.appendChild(heightEl);

        appendSVGIndicators(content, tx, ty, positionId, casteller, true);

        wrapper.draggable = true;
        wrapper.ondragstart = handleDragStart;
        wrapper.ondragend = handleDragEnd;
    } else {
        wrapper.classList.remove('filled');
        wrapper.draggable = false;
        wrapper.ondragstart = null;
        wrapper.ondragend = null;
        label.setAttribute('opacity', '1');
    }
}

export function updateCrossaSlot(positionId) {
    const wrapper = document.querySelector(`.crossa-wrapper[data-position-id="${positionId}"]`);
    if (!wrapper) return;

    const svg = wrapper.querySelector('.crossa-rect');
    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;
    const content = svg.querySelector('.slot-content');
    const label = svg.querySelector('.slot-label');

    content.innerHTML = '';

    const tx = parseFloat(label.getAttribute('x'));
    const ty = parseFloat(label.getAttribute('y'));

    if (casteller) {
        wrapper.classList.add('filled');
        label.setAttribute('opacity', '0.3');

        const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        nameEl.setAttribute('x', tx);
        nameEl.setAttribute('y', ty - 12);
        nameEl.setAttribute('text-anchor', 'middle');
        nameEl.setAttribute('class', 'slot-name');
        nameEl.textContent = casteller.name;
        content.appendChild(nameEl);

        const heightEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        heightEl.setAttribute('x', tx);
        heightEl.setAttribute('y', ty + 2);
        heightEl.setAttribute('text-anchor', 'middle');
        heightEl.setAttribute('class', 'slot-height');
        heightEl.textContent = `${casteller.height} cm`;
        content.appendChild(heightEl);

        appendSVGIndicators(content, tx, ty, positionId, casteller, true);

        wrapper.draggable = true;
        wrapper.ondragstart = handleDragStart;
        wrapper.ondragend = handleDragEnd;
    } else {
        wrapper.classList.remove('filled');
        wrapper.draggable = false;
        wrapper.ondragstart = null;
        wrapper.ondragend = null;
        label.setAttribute('opacity', '1');
    }
}

export function updateMatrixSlot(positionId) {
    const wrapper = document.querySelector(`.matrix-slot-wrapper[data-position-id="${positionId}"]`);
    if (!wrapper) return;

    const svg = wrapper.querySelector('.matrix-slot-rect');
    const name = state.getSlotContent(positionId);
    const casteller = name ? state.getCasteller(name) : null;
    const content = svg.querySelector('.slot-content');
    const label = svg.querySelector('.slot-label');

    content.innerHTML = '';

    const tx = parseFloat(label.getAttribute('x'));
    const ty = parseFloat(label.getAttribute('y'));

    if (casteller) {
        wrapper.classList.add('filled');
        label.setAttribute('opacity', '0.3');

        const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        nameEl.setAttribute('x', tx);
        nameEl.setAttribute('y', ty - 6);
        nameEl.setAttribute('text-anchor', 'middle');
        nameEl.setAttribute('class', 'slot-name');
        nameEl.textContent = casteller.name;
        content.appendChild(nameEl);

        const heightEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        heightEl.setAttribute('x', tx);
        heightEl.setAttribute('y', ty + 7);
        heightEl.setAttribute('text-anchor', 'middle');
        heightEl.setAttribute('class', 'slot-height');
        heightEl.textContent = `${casteller.height}`;
        content.appendChild(heightEl);

        wrapper.draggable = true;
        wrapper.ondragstart = handleDragStart;
        wrapper.ondragend = handleDragEnd;
    } else {
        wrapper.classList.remove('filled');
        wrapper.draggable = false;
        wrapper.ondragstart = null;
        wrapper.ondragend = null;
        label.setAttribute('opacity', '1');
    }
}
