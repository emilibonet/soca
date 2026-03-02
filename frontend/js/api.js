/**
 * Backend API integration layer.
 * 
 * Provides a clean interface for the frontend to communicate with
 * the Python backend (Flask/FastAPI). When the backend is unavailable,
 * falls back to local-only operation using in-browser state.
 * 
 * Usage:
 *   import { api } from './api.js';
 *   const castellers = await api.getCastellers();
 *   await api.runOptimization({ method: 'simulated_annealing' });
 */

export class ApiClient {
    /**
     * @param {string} baseUrl - Backend API base URL (e.g. 'http://localhost:8000/api')
     */
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this._online = false;
    }

    /** Whether the backend is reachable */
    get isOnline() {
        return this._online;
    }

    /**
     * Check if the backend is available and set online status.
     * @returns {Promise<boolean>}
     */
    async checkHealth() {
        if (!this.baseUrl) {
            this._online = false;
            return false;
        }
        try {
            const res = await fetch(`${this.baseUrl}/health`, { method: 'GET', signal: AbortSignal.timeout(3000) });
            this._online = res.ok;
        } catch {
            this._online = false;
        }
        return this._online;
    }

    // ── Castellers ────────────────────────────────────────────────

    /**
     * Fetch all castellers from the backend.
     * @returns {Promise<Array>} List of casteller objects
     */
    async getCastellers() {
        return this._get('/castellers');
    }

    /**
     * Add a new casteller.
     * @param {Object} casteller - { name, height, position_1, position_2, weight? }
     */
    async addCasteller(casteller) {
        return this._post('/castellers', casteller);
    }

    /**
     * Update an existing casteller.
     * @param {string} name - Current name (used as ID)
     * @param {Object} data - Updated casteller data
     */
    async updateCasteller(name, data) {
        return this._put(`/castellers/${encodeURIComponent(name)}`, data);
    }

    /**
     * Remove a casteller.
     * @param {string} name
     */
    async deleteCasteller(name) {
        return this._delete(`/castellers/${encodeURIComponent(name)}`);
    }

    // ── Assignments ───────────────────────────────────────────────

    /**
     * Get the current assignment state from the backend.
     * @returns {Promise<Object>} Full assignment object
     */
    async getAssignment() {
        return this._get('/assignment');
    }

    /**
     * Save the current assignment to the backend.
     * @param {Object} assignment - Full assignment object
     */
    async saveAssignment(assignment) {
        return this._post('/assignment', assignment);
    }

    /**
     * Import a full dataset (castellers + assignments) into the backend.
     * @param {Object} data - { castellers, initial_assignments, ... }
     */
    async importData(data) {
        return this._post('/assignment/import', data);
    }

    // ── Configuration ─────────────────────────────────────────────

    /**
     * Fetch configuration from the backend.
     * @returns {Promise<Object>} Configuration object
     */
    async getConfig() {
        return this._get('/config');
    }

    /**
     * Update backend configuration.
     * @param {Object} config - Partial or full config object
     */
    async updateConfig(config) {
        return this._put('/config', config);
    }

    // ── Optimization ──────────────────────────────────────────────

    /**
     * Run optimization algorithm on the backend.
     * @param {Object} params - { method, castellers, current_assignments, config }
     * @returns {Promise<Object>} Optimized assignments
     */
    async runOptimization(params) {
        return this._post('/optimize', params);
    }

    /**
     * Request auto-completion of empty positions.
     * @param {Object} params - { castellers, current_assignments, config }
     * @returns {Promise<Object>} Completed assignments
     */
    async autoComplete(params) {
        return this._post('/auto-complete', params);
    }

    /**
     * Score a casteller for a specific position.
     * @param {Object} params - { casteller, position_id, assignments, config }
     * @returns {Promise<Object>} { score, breakdown }
     */
    async scoreCasteller(params) {
        return this._post('/score', params);
    }

    // ── HTTP Helpers ──────────────────────────────────────────────

    async _get(path) {
        return this._request('GET', path);
    }

    async _post(path, body) {
        return this._request('POST', path, body);
    }

    async _put(path, body) {
        return this._request('PUT', path, body);
    }

    async _delete(path) {
        return this._request('DELETE', path);
    }

    async _request(method, path, body = null) {
        if (!this.baseUrl) {
            throw new Error('API base URL not configured. Running in local-only mode.');
        }

        const options = {
            method,
            headers: { 'Content-Type': 'application/json' },
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`${this.baseUrl}${path}`, options);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new ApiError(response.status, errorData.detail || response.statusText, errorData);
        }

        // 204 No Content
        if (response.status === 204) return null;

        return response.json();
    }
}

/**
 * Structured API error with status code and server details.
 */
export class ApiError extends Error {
    constructor(status, message, data = {}) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
    }
}

/**
 * Singleton API client instance.
 * Configure the base URL before use:
 *   api.baseUrl = 'http://localhost:8000/api';
 */
export const api = new ApiClient();
