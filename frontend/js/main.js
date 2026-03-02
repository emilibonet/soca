/**
 * Application entry point.
 * Wires up all event listeners and initializes the application.
 */

import { state } from './state.js';
import { initPinya } from './pinya.js';
import { updatePool, setPoolCallbacks } from './pool.js';
import { updateStatusBar } from './statusBar.js';
import { initSettings } from './settings.js';
import { initTroncPanel } from './troncPanel.js';
import { autoComplete } from './algorithms.js';
import { updateAllSlots } from './ui.js';
import {
    zoomIn, zoomOut, zoomReset,
    initCanvasPanZoom,
    undo, redo,
    updateHistoryButtons,
    exportAssignment, importAssignment, handleFileImport,
    openCastellerModal, closeModal, saveCasteller,
    editCasteller, removeCasteller,
    setControlsUpdateFn
} from './controls.js';

// ── Initialization ────────────────────────────────────────────

function init() {
    // Wire up callback injection to break circular dependencies
    setPoolCallbacks({ editCasteller, removeCasteller });
    setControlsUpdateFn(updateAllSlots);

    // Load data
    state.loadSampleData();

    // Initialize panels
    initSettings();
    initTroncPanel();
    initPinya(updateAllSlots);

    // Initial renders
    updatePool();
    updateStatusBar();
    updateHistoryButtons();

    // Pan & zoom
    initCanvasPanZoom();

    // ── Button Event Listeners ────────────────────────────────
    document.getElementById('autoCompleteBtn').addEventListener('click', autoComplete);
    document.getElementById('loadBtn').addEventListener('click', importAssignment);
    document.getElementById('saveBtn').addEventListener('click', exportAssignment);
    document.getElementById('undoBtn').addEventListener('click', undo);
    document.getElementById('redoBtn').addEventListener('click', redo);
    document.getElementById('addCastellerBtn').addEventListener('click', () => openCastellerModal());
    document.getElementById('zoomInBtn').addEventListener('click', zoomIn);
    document.getElementById('zoomOutBtn').addEventListener('click', zoomOut);
    document.getElementById('zoomResetBtn').addEventListener('click', zoomReset);
    document.getElementById('poolSearch').addEventListener('input', updatePool);
    document.getElementById('poolSort').addEventListener('change', updatePool);
    document.getElementById('castellerForm').addEventListener('submit', saveCasteller);
    document.getElementById('fileInput').addEventListener('change', handleFileImport);

    // Close modal buttons
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    const modalCancelBtn = document.getElementById('modalCancelBtn');
    if (modalCancelBtn) modalCancelBtn.addEventListener('click', closeModal);

    // Close modal on backdrop click
    const modal = document.getElementById('castellerModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
    }

    // ── Keyboard Shortcuts ────────────────────────────────────
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') {
                e.preventDefault();
                undo();
            } else if (e.key === 'y') {
                e.preventDefault();
                redo();
            } else if (e.key === 's') {
                e.preventDefault();
                exportAssignment();
            } else if (e.key === 'f') {
                e.preventDefault();
                document.getElementById('poolSearch').focus();
            }
        }
    });
}

window.addEventListener('DOMContentLoaded', init);
