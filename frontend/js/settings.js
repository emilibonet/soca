/**
 * Settings panel — position and queue configuration UI.
 */

import { state } from './state.js';
import { bus, Events } from './events.js';

// ── Init ──────────────────────────────────────────────────────

export function initSettings() {
    const container = document.getElementById('settingsContent');

    // Global settings
    container.innerHTML = `
        <div class="settings-section expanded">
            <h3>Global</h3>
            <div class="settings-content">
                <div class="setting-row">
                    <label>Optimization Method</label>
                    <select id="optMethod">
                        <option value="greedy">Greedy</option>
                        <option value="exhaustive">Exhaustive</option>
                        <option value="simulated_annealing">Simulated Annealing</option>
                        <option value="adaptive_simulated_annealing">Adaptive Simulated Annealing</option>
                    </select>
                </div>
                <div class="setting-row">
                    <label>
                        <input type="checkbox" id="useWeight"> Use Weight Data
                    </label>
                </div>
            </div>
        </div>
    `;

    // Position settings
    ['baix', 'crossa', 'contrafort', 'agulla'].forEach(pos => {
        const section = createPositionSettings(pos);
        container.appendChild(section);
    });

    // Queue settings
    ['mans', 'daus', 'laterals'].forEach(q => {
        const section = createQueueSettings(q);
        container.appendChild(section);
    });

    // Toggle functionality
    container.querySelectorAll('.settings-section h3').forEach(h3 => {
        h3.addEventListener('click', () => {
            h3.parentElement.classList.toggle('expanded');
        });
    });

    // Load current values
    loadSettings();
}

// ── Position Settings Section ─────────────────────────────────

function createPositionSettings(posName) {
    const config = state.config.positions[posName];
    const section = document.createElement('div');
    section.className = 'settings-section';
    section.innerHTML = `
        <h3>${posName.charAt(0).toUpperCase() + posName.slice(1)}</h3>
        <div class="settings-content">
            <div class="setting-row">
                <label>Height Ratio Min</label>
                <input type="number" step="0.01" data-config="positions.${posName}.height_ratio_min" value="${config.height_ratio_min}">
            </div>
            <div class="setting-row">
                <label>Height Ratio Max</label>
                <input type="number" step="0.01" data-config="positions.${posName}.height_ratio_max" value="${config.height_ratio_max}">
            </div>
            <div class="setting-row">
                <label>Height Weight</label>
                <input type="range" min="0" max="1" step="0.1" data-config="positions.${posName}.height_weight" value="${config.height_weight}">
                <span class="range-value">${config.height_weight}</span>
            </div>
            <div class="setting-row">
                <label>Expertise Weight</label>
                <input type="range" min="0" max="1" step="0.1" data-config="positions.${posName}.expertise_weight" value="${config.expertise_weight}">
                <span class="range-value">${config.expertise_weight}</span>
            </div>
        </div>
    `;

    section.querySelectorAll('input[type="range"]').forEach(input => {
        input.addEventListener('input', (e) => {
            const valueSpan = e.target.nextElementSibling;
            if (valueSpan) valueSpan.textContent = e.target.value;
            updateConfigValue(e.target.dataset.config, parseFloat(e.target.value));
        });
    });

    section.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('change', (e) => {
            updateConfigValue(e.target.dataset.config, parseFloat(e.target.value));
        });
    });

    return section;
}

// ── Queue Settings Section ────────────────────────────────────

function createQueueSettings(queueName) {
    const config = state.config.queues[queueName];
    const section = document.createElement('div');
    section.className = 'settings-section';
    section.innerHTML = `
        <h3>${queueName.charAt(0).toUpperCase() + queueName.slice(1)}</h3>
        <div class="settings-content">
            <div class="setting-row">
                <label>Max Depth</label>
                <input type="number" min="1" max="20" data-config="queues.${queueName}.max_depth" value="${config.max_depth}">
            </div>
            <div class="setting-row">
                <label>Height Ratio Min (depth 1)</label>
                <input type="number" step="0.01" data-config="queues.${queueName}.height_ratio_min" value="${config.height_ratio_min}">
            </div>
            <div class="setting-row">
                <label>Height Ratio Max (depth 1)</label>
                <input type="number" step="0.01" data-config="queues.${queueName}.height_ratio_max" value="${config.height_ratio_max}">
            </div>
            <div class="setting-row">
                <label>Queue Height Ratio Min (depth 2+)</label>
                <input type="number" step="0.01" data-config="queues.${queueName}.queue_height_ratio_min" value="${config.queue_height_ratio_min}">
            </div>
            <div class="setting-row">
                <label>Queue Height Ratio Max (depth 2+)</label>
                <input type="number" step="0.01" data-config="queues.${queueName}.queue_height_ratio_max" value="${config.queue_height_ratio_max}">
            </div>
        </div>
    `;

    section.querySelectorAll('input').forEach(input => {
        input.addEventListener('change', (e) => {
            updateConfigValue(e.target.dataset.config, parseFloat(e.target.value));
        });
    });

    return section;
}

// ── Load / Update ─────────────────────────────────────────────

function loadSettings() {
    document.getElementById('optMethod').value = state.config.optimization.method;
    document.getElementById('useWeight').checked = state.config.optimization.use_weight;

    document.getElementById('optMethod').addEventListener('change', (e) => {
        state.config.optimization.method = e.target.value;
    });

    document.getElementById('useWeight').addEventListener('change', (e) => {
        state.config.optimization.use_weight = e.target.checked;
    });
}

function updateConfigValue(path, value) {
    const parts = path.split('.');
    let obj = state.config;
    for (let i = 0; i < parts.length - 1; i++) {
        obj = obj[parts[i]];
    }
    obj[parts[parts.length - 1]] = value;

    // Update indicators as ranges may have changed
    bus.emit(Events.UI_REFRESH);
}
