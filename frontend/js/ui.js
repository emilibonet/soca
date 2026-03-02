/**
 * Central UI coordinator — refreshes all visual components.
 * Subscribes to bus events so modules can trigger updates without circular deps.
 */

import { updateSlot, updateAgullaSlot, updateBaixSlot, updateContrafortSlot, updateCrossaSlot } from './slotUpdater.js';
import { rebuildMansQueues, rebuildMatrixQueues } from './queues.js';
import { resolveCrossaOverlaps } from './crossa.js';
import { updateTroncPanel } from './troncPanel.js';
import { updatePool } from './pool.js';
import { updateStatusBar } from './statusBar.js';
import { updateHistoryButtons } from './controls.js';
import { bus, Events } from './events.js';

// ── Full Slot Refresh ─────────────────────────────────────────

export function updateAllSlots() {
    // HTML-based slots
    document.querySelectorAll('.slot').forEach(slot => {
        const posId = slot.dataset.positionId;
        updateSlot(posId);
    });
    // SVG agulla triangles
    document.querySelectorAll('.agulla-wrapper').forEach(wrapper => {
        updateAgullaSlot(wrapper.dataset.positionId);
    });
    // SVG baix rectangles (synced with tronc panel)
    document.querySelectorAll('.baix-wrapper').forEach(wrapper => {
        updateBaixSlot(wrapper.dataset.positionId);
    });
    // SVG contrafort rectangles
    document.querySelectorAll('.contrafort-wrapper').forEach(wrapper => {
        updateContrafortSlot(wrapper.dataset.positionId);
    });
    // SVG crossa rectangles
    document.querySelectorAll('.crossa-wrapper').forEach(wrapper => {
        updateCrossaSlot(wrapper.dataset.positionId);
    });
    // Dynamic mans SVG queue
    rebuildMansQueues();
    // Dynamic matrix queues (laterals + daus)
    rebuildMatrixQueues();
    // Resolve overlapping crossa rectangles
    resolveCrossaOverlaps();
    // Tronc & Pom panel (right sidebar)
    updateTroncPanel();
}

// ── Event Subscriptions ───────────────────────────────────────

bus.on(Events.UI_REFRESH, () => {
    updateAllSlots();
    updatePool();
    updateStatusBar();
    updateHistoryButtons();
});

bus.on(Events.HISTORY_CHANGED, () => {
    updateHistoryButtons();
});
