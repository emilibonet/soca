/**
 * Status bar — displays validation checks and height compliance.
 */

import { state } from './state.js';
import { calculateHeightRatio, getHeightRatioRange } from './slotUpdater.js';

export function updateStatusBar() {
    const statusBar = document.getElementById('statusBar');
    const checks = [];

    // Check agulles filled
    const agulles = ['agulla-Rengla', 'agulla-Plena', 'agulla-Buida'];
    const agullesFilled = agulles.every(id => state.getSlotContent(id));
    checks.push({
        label: 'All agulles filled',
        valid: agullesFilled,
        critical: true
    });

    // Check baix filled
    const baixs = ['baix-Rengla', 'baix-Plena', 'baix-Buida'];
    const baixsFilled = baixs.every(id => state.getSlotContent(id));
    checks.push({
        label: 'Baix in all columns',
        valid: baixsFilled,
        critical: true
    });

    // Check mans depth 1
    const mansQueues = ['Rengla', 'Plena', 'Buida'];
    const mansD1Filled = mansQueues.every(q => state.getSlotContent(`mans-${q}-0`));
    checks.push({
        label: 'Mans depth 1 in all queues',
        valid: mansD1Filled,
        critical: true
    });

    // Calculate height compliance
    let totalSlots = 0;
    let compliantSlots = 0;
    document.querySelectorAll('.slot.filled:not(.tronc), .matrix-slot-wrapper.filled').forEach(slot => {
        const posId = slot.dataset.positionId;
        const name = state.getSlotContent(posId);
        if (!name) return;

        const casteller = state.getCasteller(name);
        if (!casteller) return;

        totalSlots++;
        const ratio = calculateHeightRatio(posId, casteller);
        if (ratio !== null) {
            const [posType,,idx] = state.parsePositionId(posId);
            const { min, max } = getHeightRatioRange(posType, idx || 0);
            if (min && max && ratio >= min && ratio <= max) {
                compliantSlots++;
            }
        }
    });

    const heightCompliance = totalSlots > 0 ? (compliantSlots / totalSlots * 100).toFixed(0) : 0;
    checks.push({
        label: `Height compliance: ${heightCompliance}%`,
        valid: heightCompliance >= 70,
        critical: false
    });

    // Render status items
    statusBar.innerHTML = checks.map(check => {
        const className = check.critical
            ? (check.valid ? 'valid' : 'invalid')
            : (check.valid ? 'valid' : 'warning');
        const icon = check.valid ? '\u2705' : (check.critical ? '\u274C' : '\u26A0\uFE0F');
        return `<div class="status-item ${className}">${icon} ${check.label}</div>`;
    }).join('');
}
