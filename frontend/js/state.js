/**
 * Central application state management.
 * Holds castellers, assignments, config, undo/redo history, and view state (zoom/pan).
 */

import { DEFAULT_CONFIG } from './config.js';
import { SAMPLE_DATA } from './sampleData.js';
import { bus, Events } from './events.js';

/** Map from CSS-safe position ID prefixes to actual assignment keys */
const TYPE_MAP = { 'terc': 'terç' };

export class AppState {
    constructor() {
        this.castellers = [];
        this.assignments = {
            enxaneta: {},
            acotxador: {},
            dosos: {},
            baix: {},
            segon: {},
            terç: {},
            crossa: {},
            contrafort: {},
            agulla: {},
            mans: {},
            daus: {},
            laterals: {}
        };
        this.config = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
        this.history = [];
        this.historyIndex = -1;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.draggedElement = null;
        this.draggedCasteller = null;
        this.dragSourcePosId = null;
    }

    // ── Data Loading ──────────────────────────────────────────────

    loadSampleData() {
        this.castellers = JSON.parse(JSON.stringify(SAMPLE_DATA.castellers));
        this.assignments = JSON.parse(JSON.stringify(SAMPLE_DATA.initial_assignments));
        this.saveHistory('Initial load');
    }

    // ── Casteller Lookups ─────────────────────────────────────────

    getCasteller(name) {
        return this.castellers.find(c => c.name === name);
    }

    getAssignedNames() {
        const names = new Set();
        for (const [pos, cols] of Object.entries(this.assignments)) {
            for (const val of Object.values(cols)) {
                if (Array.isArray(val)) {
                    if (val.length > 0 && Array.isArray(val[0])) {
                        val.forEach(depth => { if (depth[0]) names.add(depth[0]); });
                    } else {
                        val.forEach(name => { if (name) names.add(name); });
                    }
                }
            }
        }
        return names;
    }

    getUnassigned() {
        const assigned = this.getAssignedNames();
        return this.castellers.filter(c => !assigned.has(c.name));
    }

    // ── Assignment Operations ─────────────────────────────────────

    assignToSlot(positionId, name) {
        const [posType, col, idx] = this.parsePositionId(positionId);
        this.removeFromAssignments(name);

        if (['mans', 'daus', 'laterals'].includes(posType)) {
            if (!this.assignments[posType][col]) {
                this.assignments[posType][col] = [];
            }
            while (this.assignments[posType][col].length <= idx) {
                this.assignments[posType][col].push([null]);
            }
            this.assignments[posType][col][idx] = [name];
        } else {
            if (!this.assignments[posType][col]) {
                this.assignments[posType][col] = [];
            }
            if (idx !== undefined) {
                while (this.assignments[posType][col].length <= idx) {
                    this.assignments[posType][col].push(null);
                }
                this.assignments[posType][col][idx] = name;
            } else {
                this.assignments[posType][col] = [name];
            }
        }
    }

    removeFromSlot(positionId) {
        const [posType, col, idx] = this.parsePositionId(positionId);

        if (['mans', 'daus', 'laterals'].includes(posType)) {
            if (this.assignments[posType]?.[col] && idx < this.assignments[posType][col].length) {
                this.assignments[posType][col].splice(idx, 1);
            }
        } else {
            if (this.assignments[posType][col]) {
                if (idx !== undefined) {
                    this.assignments[posType][col][idx] = null;
                } else {
                    this.assignments[posType][col] = [];
                }
            }
        }
    }

    removeFromAssignments(name) {
        for (const [posType, cols] of Object.entries(this.assignments)) {
            for (const [col, val] of Object.entries(cols)) {
                if (Array.isArray(val)) {
                    if (val.length > 0 && Array.isArray(val[0])) {
                        for (let i = val.length - 1; i >= 0; i--) {
                            if (val[i][0] === name) val.splice(i, 1);
                        }
                    } else {
                        for (let i = 0; i < val.length; i++) {
                            if (val[i] === name) val[i] = null;
                        }
                    }
                }
            }
        }
    }

    getSlotContent(positionId) {
        const [posType, col, idx] = this.parsePositionId(positionId);

        if (!this.assignments[posType]) return null;

        if (['mans', 'daus', 'laterals'].includes(posType)) {
            if (this.assignments[posType][col] && this.assignments[posType][col][idx]) {
                return this.assignments[posType][col][idx][0];
            }
        } else {
            if (this.assignments[posType][col]) {
                if (idx !== undefined) {
                    return this.assignments[posType][col][idx];
                } else {
                    return this.assignments[posType][col][0];
                }
            }
        }
        return null;
    }

    /**
     * Parse a position ID string into [posType, col, idx].
     * Format: "posType-col-idx" or "posType-col"
     */
    parsePositionId(id) {
        const parts = id.split('-');
        let posType = parts[0];
        posType = TYPE_MAP[posType] || posType;
        const col = parts.slice(1, -1).join('-') || parts[1];
        const idx = parts.length > 2 && !isNaN(parts[parts.length - 1])
            ? parseInt(parts[parts.length - 1])
            : undefined;
        return [posType, col, idx];
    }

    // ── Casteller CRUD ────────────────────────────────────────────

    addCasteller(casteller) {
        if (this.castellers.find(c => c.name === casteller.name)) {
            throw new Error('Casteller with this name already exists');
        }
        this.castellers.push(casteller);
        this.saveHistory(`Added ${casteller.name}`);
    }

    removeCasteller(name) {
        this.removeFromAssignments(name);
        this.castellers = this.castellers.filter(c => c.name !== name);
        this.saveHistory(`Removed ${name}`);
    }

    updateCasteller(oldName, newCasteller) {
        const idx = this.castellers.findIndex(c => c.name === oldName);
        if (idx === -1) return;

        if (oldName !== newCasteller.name) {
            for (const [posType, cols] of Object.entries(this.assignments)) {
                for (const [col, val] of Object.entries(cols)) {
                    if (Array.isArray(val)) {
                        if (val.length > 0 && Array.isArray(val[0])) {
                            for (let i = 0; i < val.length; i++) {
                                if (val[i][0] === oldName) val[i] = [newCasteller.name];
                            }
                        } else {
                            for (let i = 0; i < val.length; i++) {
                                if (val[i] === oldName) val[i] = newCasteller.name;
                            }
                        }
                    }
                }
            }
        }

        this.castellers[idx] = newCasteller;
        this.saveHistory(`Updated ${newCasteller.name}`);
    }

    // ── History (Undo / Redo) ─────────────────────────────────────

    saveHistory(action) {
        this.history = this.history.slice(0, this.historyIndex + 1);

        this.history.push({
            action,
            state: {
                castellers: JSON.parse(JSON.stringify(this.castellers)),
                assignments: JSON.parse(JSON.stringify(this.assignments))
            }
        });

        this.historyIndex++;

        if (this.history.length > 50) {
            this.history.shift();
            this.historyIndex--;
        }

        bus.emit(Events.HISTORY_CHANGED);
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const snapshot = this.history[this.historyIndex];
            this.castellers = JSON.parse(JSON.stringify(snapshot.state.castellers));
            this.assignments = JSON.parse(JSON.stringify(snapshot.state.assignments));
            return true;
        }
        return false;
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            const snapshot = this.history[this.historyIndex];
            this.castellers = JSON.parse(JSON.stringify(snapshot.state.castellers));
            this.assignments = JSON.parse(JSON.stringify(snapshot.state.assignments));
            return true;
        }
        return false;
    }

    // ── Import / Export ───────────────────────────────────────────

    exportJSON() {
        const assigned = this.getAssignedNames();
        const unassigned = this.castellers.filter(c => !assigned.has(c.name));

        return {
            assignments: this.assignments,
            summary: {
                total_assigned: assigned.size,
                unassigned: unassigned
            },
            configuration: {
                columns: ['Rengla', 'Plena', 'Buida'],
                tronc_positions: ['baix', 'segon', 'terç'],
                optimization_method: this.config.optimization.method,
                use_weight: this.config.optimization.use_weight
            }
        };
    }

    importJSON(data) {
        if (data.castellers) {
            this.castellers = data.castellers;
        }
        if (data.initial_assignments || data.assignments) {
            this.assignments = data.initial_assignments || data.assignments;
        }
        if (data.configuration) {
            if (data.configuration.optimization_method) {
                this.config.optimization.method = data.configuration.optimization_method;
            }
            if (data.configuration.use_weight !== undefined) {
                this.config.optimization.use_weight = data.configuration.use_weight;
            }
        }
        this.saveHistory('Imported data');
    }
}

/** Singleton application state instance */
export const state = new AppState();
