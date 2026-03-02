/**
 * Pinya canvas initialization — creates all position slot elements.
 * Agulla, baix, contrafort, crossa SVG wrappers.
 * Mans and matrix queues are created dynamically by queues.js.
 */

import { handleDragOver, handleDrop, handleDragLeave } from './dragDrop.js';
import { showSmartSuggestions } from './algorithms.js';

// ── Main Init ─────────────────────────────────────────────────

export function initPinya(updateAllSlots) {
    const container = document.getElementById('canvasInner');
    container.innerHTML = '';

    createSlots(container);
    updateAllSlots();
}

// ── Slot Creation ─────────────────────────────────────────────

function createSlots(container) {
    // Triangle guide
    const triangle = document.createElement('div');
    triangle.className = 'tronc-triangle';
    container.appendChild(triangle);

    // Column labels
    [
        { text: 'R', class: 'label-rengla' },
        { text: 'P', class: 'label-plena' },
        { text: 'B', class: 'label-buida' }
    ].forEach(cfg => {
        const label = document.createElement('div');
        label.className = `column-label ${cfg.class}`;
        label.textContent = cfg.text;
        container.appendChild(label);
    });

    // No HTML-based position slots remain (all are SVG wrappers now)
    const positions = [];
    positions.forEach(pos => {
        const slot = createSlot(pos);
        container.appendChild(slot);
    });

    // ── AGULLA — SVG triangles ────────────────────────────────
    const agullaConfigs = [
        {
            id: 'agulla-Rengla', col: 'Rengla', label: 'Agulla R',
            cls: 'agulla-rengla col-rengla',
            viewBox: '0 0 160 47', w: 160, h: 47,
            points: '80,0 0,47 160,47',
            stroke: 'var(--rengla)',
            textX: 80, textY: 35
        },
        {
            id: 'agulla-Buida', col: 'Buida', label: 'Agulla B',
            cls: 'agulla-buida col-buida',
            viewBox: '0 0 80 139', w: 80, h: 139,
            points: '80,93 80,0 0,139',
            stroke: 'var(--buida)',
            textX: 42, textY: 85
        },
        {
            id: 'agulla-Plena', col: 'Plena', label: 'Agulla P',
            cls: 'agulla-plena col-plena',
            viewBox: '0 0 80 139', w: 80, h: 139,
            points: '0,93 0,0 80,139',
            stroke: 'var(--plena)',
            textX: 38, textY: 85
        }
    ];
    agullaConfigs.forEach(cfg => createSVGWrapper(container, cfg, 'agulla-wrapper', 'agulla-triangle', 'agulla'));

    // ── BAIX — SVG rectangles ─────────────────────────────────
    const baixConfigs = [
        {
            id: 'baix-Rengla', col: 'Rengla', label: 'Baix R',
            cls: 'baix-rengla col-rengla',
            viewBox: '0 0 120 55',
            points: '0,0 120,0 120,55 0,55',
            stroke: 'var(--rengla)',
            textX: 60, textY: 35
        },
        {
            id: 'baix-Buida', col: 'Buida', label: 'Baix B',
            cls: 'baix-buida col-buida',
            viewBox: '0 0 108 132',
            points: '48,132 108,27 60,0 0,104',
            stroke: 'var(--buida)',
            textX: 54, textY: 66
        },
        {
            id: 'baix-Plena', col: 'Plena', label: 'Baix P',
            cls: 'baix-plena col-plena',
            viewBox: '0 0 108 132',
            points: '60,132 0,27 48,0 108,104',
            stroke: 'var(--plena)',
            textX: 54, textY: 66
        }
    ];
    baixConfigs.forEach(cfg => createSVGWrapper(container, cfg, 'baix-wrapper', 'baix-rect', 'baix'));

    // ── CONTRAFORT — SVG rectangles ───────────────────────────
    const contrafortConfigs = [
        {
            id: 'contrafort-Rengla', col: 'Rengla', label: 'Ctfort R',
            cls: 'contrafort-rengla col-rengla',
            viewBox: '0 0 120 55',
            points: '0,0 120,0 120,55 0,55',
            stroke: 'var(--rengla)',
            textX: 60, textY: 35
        },
        {
            id: 'contrafort-Buida', col: 'Buida', label: 'Ctfort B',
            cls: 'contrafort-buida col-buida',
            viewBox: '0 0 108 133',
            points: '47,132 107,28 60,1 0,105',
            stroke: 'var(--buida)',
            textX: 54, textY: 67
        },
        {
            id: 'contrafort-Plena', col: 'Plena', label: 'Ctfort P',
            cls: 'contrafort-plena col-plena',
            viewBox: '0 0 108 133',
            points: '61,132 1,28 48,1 108,105',
            stroke: 'var(--plena)',
            textX: 54, textY: 67
        }
    ];
    contrafortConfigs.forEach(cfg => createSVGWrapper(container, cfg, 'contrafort-wrapper', 'contrafort-rect', 'contrafort'));

    // ── CROSSA — SVG rectangles ───────────────────────────────
    const crossaConfigs = [
        {
            id: 'crossa-Rengla-0', col: 'Rengla', label: 'Crossa R-1',
            cls: 'crossa-rengla-0 col-rengla',
            viewBox: '0 0 50 110',
            points: '0,0 50,0 50,110 0,110',
            stroke: 'var(--rengla)',
            textX: 25, textY: 55
        },
        {
            id: 'crossa-Rengla-1', col: 'Rengla', label: 'Crossa R-2',
            cls: 'crossa-rengla-1 col-rengla',
            viewBox: '0 0 50 110',
            points: '0,0 50,0 50,110 0,110',
            stroke: 'var(--rengla)',
            textX: 25, textY: 55
        },
        {
            id: 'crossa-Buida-0', col: 'Buida', label: 'Crossa B-1',
            cls: 'crossa-buida-0 col-buida',
            viewBox: '0 0 120 97',
            points: '120,54 25,0 0,43 95,97',
            stroke: 'var(--buida)',
            textX: 60, textY: 49
        },
        {
            id: 'crossa-Buida-1', col: 'Buida', label: 'Crossa B-2',
            cls: 'crossa-buida-1 col-buida',
            viewBox: '0 0 120 98',
            points: '95,98 0,43 25,0 120,55',
            stroke: 'var(--buida)',
            textX: 60, textY: 49
        },
        {
            id: 'crossa-Plena-0', col: 'Plena', label: 'Crossa P-1',
            cls: 'crossa-plena-0 col-plena',
            viewBox: '0 0 120 97',
            points: '0,54 95,0 120,43 25,97',
            stroke: 'var(--plena)',
            textX: 60, textY: 49
        },
        {
            id: 'crossa-Plena-1', col: 'Plena', label: 'Crossa P-2',
            cls: 'crossa-plena-1 col-plena',
            viewBox: '0 0 120 98',
            points: '25,98 120,43 95,0 0,55',
            stroke: 'var(--plena)',
            textX: 60, textY: 49
        }
    ];
    crossaConfigs.forEach(cfg => createSVGWrapper(container, cfg, 'crossa-wrapper', 'crossa-rect', 'crossa'));

    // Mans queues are created dynamically by rebuildMansQueues() in updateAllSlots()
    // Matrix queues (laterals + daus) are created dynamically by rebuildMatrixQueues()
}

// ── SVG Wrapper Factory ───────────────────────────────────────

function createSVGWrapper(container, cfg, wrapperClass, svgClass, posType) {
    const wrapper = document.createElement('div');
    wrapper.className = `${wrapperClass} ${cfg.cls}`;
    wrapper.dataset.positionId = cfg.id;
    wrapper.dataset.positionType = posType;
    wrapper.dataset.readOnly = 'false';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', svgClass);
    svg.setAttribute('viewBox', cfg.viewBox);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');

    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', cfg.points);
    poly.setAttribute('fill', 'var(--bg-card)');
    poly.setAttribute('stroke', cfg.stroke);
    poly.setAttribute('stroke-width', '2');
    svg.appendChild(poly);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', cfg.textX);
    label.setAttribute('y', cfg.textY);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('class', 'slot-label');
    label.textContent = cfg.label;
    svg.appendChild(label);

    const content = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    content.setAttribute('class', 'slot-content');
    svg.appendChild(content);

    wrapper.appendChild(svg);

    wrapper.addEventListener('dragover', handleDragOver);
    wrapper.addEventListener('drop', handleDrop);
    wrapper.addEventListener('dragleave', handleDragLeave);
    wrapper.addEventListener('click', () => showSmartSuggestions(cfg.id));

    container.appendChild(wrapper);
}

// ── HTML Slot Factory (for any future non-SVG slots) ──────────

function createSlot(config) {
    const slot = document.createElement('div');
    slot.className = `slot ${config.class}`;
    slot.dataset.positionId = config.id;
    slot.dataset.positionType = config.type;
    slot.dataset.readOnly = config.readOnly || false;

    if (!config.readOnly) {
        slot.addEventListener('dragover', handleDragOver);
        slot.addEventListener('drop', handleDrop);
        slot.addEventListener('dragleave', handleDragLeave);
        slot.addEventListener('click', () => showSmartSuggestions(config.id));
    }

    const label = document.createElement('div');
    label.className = 'slot-label';
    label.textContent = config.label;
    slot.appendChild(label);

    return slot;
}
