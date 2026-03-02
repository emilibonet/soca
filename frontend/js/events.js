/**
 * Lightweight event bus for decoupled module communication.
 * Avoids circular dependencies between modules that need to
 * trigger UI refreshes or notify each other of state changes.
 */
class EventBus {
    constructor() {
        this._listeners = {};
    }

    /**
     * Subscribe to an event.
     * @param {string} event - Event name
     * @param {Function} fn  - Callback
     * @returns {Function} Unsubscribe function
     */
    on(event, fn) {
        (this._listeners[event] ||= []).push(fn);
        return () => this.off(event, fn);
    }

    /**
     * Unsubscribe from an event.
     */
    off(event, fn) {
        const list = this._listeners[event];
        if (list) {
            this._listeners[event] = list.filter(f => f !== fn);
        }
    }

    /**
     * Emit an event with optional arguments.
     */
    emit(event, ...args) {
        (this._listeners[event] || []).forEach(fn => fn(...args));
    }
}

export const bus = new EventBus();

// Standard event names used across the application
export const Events = {
    /** Fired after any assignment / state change that requires full UI refresh */
    UI_REFRESH: 'ui:refresh',
    /** Fired when history changes (undo/redo availability) */
    HISTORY_CHANGED: 'state:history-changed',
    /** Fired when data is loaded from API or import */
    DATA_LOADED: 'data:loaded',
};
