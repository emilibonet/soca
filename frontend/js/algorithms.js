/**
 * Auto-complete algorithm, casteller scoring, and smart suggestions.
 */

import { state } from './state.js';
import { bus, Events } from './events.js';
import { showToast } from './toast.js';
import {
    getPositionKeywords,
    getReferenceHeight,
    getHeightRatioRange
} from './slotUpdater.js';

// ── Auto-Complete ─────────────────────────────────────────────

export function autoComplete() {
    let assigned = 0;
    const unassigned = state.getUnassigned();

    if (unassigned.length === 0) {
        showToast('All castellers are already assigned');
        return;
    }

    // Get all empty slots in priority order
    const emptySlots = getEmptySlotsByPriority();

    for (const slotId of emptySlots) {
        // Get available candidates
        const available = state.getUnassigned();
        if (available.length === 0) break;

        // Score each candidate
        const scored = available.map(c => ({
            casteller: c,
            score: scoreCasteller(slotId, c)
        }));

        // Sort by score (highest first)
        scored.sort((a, b) => b.score - a.score);

        // Assign the best one
        if (scored.length > 0 && scored[0].score > 0) {
            state.assignToSlot(slotId, scored[0].casteller.name);
            assigned++;
        }
    }

    bus.emit(Events.UI_REFRESH);
    state.saveHistory(`Auto-completed ${assigned} positions`);
    showToast(`Auto-assigned ${assigned} castellers. ${state.getUnassigned().length} remain unassigned.`);
}

// ── Empty Slots by Priority ───────────────────────────────────

function getEmptySlotsByPriority() {
    const slots = [];

    // Priority order: baix → crossa → contrafort → agulla → mans → daus → laterals
    const positions = [
        'baix-Rengla', 'baix-Plena', 'baix-Buida',
        'crossa-Rengla-0', 'crossa-Rengla-1', 'crossa-Plena-0', 'crossa-Plena-1', 'crossa-Buida-0', 'crossa-Buida-1',
        'contrafort-Rengla', 'contrafort-Plena', 'contrafort-Buida',
        'agulla-Rengla', 'agulla-Plena', 'agulla-Buida'
    ];

    for (const posId of positions) {
        if (!state.getSlotContent(posId)) {
            slots.push(posId);
        }
    }

    // Queues — breadth first
    const queueTypes = ['mans', 'daus', 'laterals'];
    for (const qType of queueTypes) {
        const maxDepth = state.config.queues[qType]?.max_depth || 4;
        for (let d = 0; d < maxDepth; d++) {
            const queues = Object.keys(state.assignments[qType] || {});
            for (const qId of queues) {
                const posId = `${qType}-${qId}-${d}`;
                if (!state.getSlotContent(posId)) {
                    slots.push(posId);
                }
            }
        }
    }

    return slots;
}

// ── Casteller Scoring ─────────────────────────────────────────

export function scoreCasteller(slotId, casteller) {
    const [posType, col, idx] = state.parsePositionId(slotId);

    let score = 0;

    // Expertise score (0-1.0)
    const keywords = getPositionKeywords(posType);
    const pos1 = (casteller.position_1 || '').toLowerCase();
    const pos2 = (casteller.position_2 || '').toLowerCase();

    let expertiseScore = 0.1; // baseline
    keywords.forEach(kw => {
        if (pos1.includes(kw.toLowerCase())) expertiseScore = 1.0;
        else if (pos2.includes(kw.toLowerCase())) expertiseScore = Math.max(expertiseScore, 0.5);
    });

    // Height score (0-1.0)
    let heightScore = 0.5;
    const refHeight = getReferenceHeight(slotId);
    if (refHeight) {
        const ratio = casteller.height / refHeight;
        const { min, max } = getHeightRatioRange(posType, idx || 0);
        if (min && max) {
            if (ratio >= min && ratio <= max) {
                heightScore = 1.0;
            } else {
                const distance = ratio < min ? min - ratio : ratio - max;
                heightScore = Math.max(0, 1.0 - distance * 2);
            }
        }
    }

    // Get weights from config
    const config = ['mans', 'daus', 'laterals'].includes(posType)
        ? state.config.queues[posType]
        : state.config.positions[posType];

    if (config) {
        score = (config.expertise_weight || 0.5) * expertiseScore +
                (config.height_weight || 0.5) * heightScore;
    } else {
        score = 0.5 * expertiseScore + 0.5 * heightScore;
    }

    return score;
}

// ── Smart Suggestions ─────────────────────────────────────────

export function showSmartSuggestions(positionId) {
    const available = state.getUnassigned();
    if (available.length === 0) return;

    // Score all available castellers
    const scored = available.map(c => ({
        casteller: c,
        score: scoreCasteller(positionId, c)
    }));

    // Sort and take top 5
    scored.sort((a, b) => b.score - a.score);
    const top5 = scored.slice(0, 5);

    // Simple prompt-based menu
    const message = top5.map((s, i) =>
        `${i + 1}. ${s.casteller.name} (${s.casteller.height}cm) - Score: ${s.score.toFixed(2)}`
    ).join('\n');

    const choice = prompt(`Suggestions for ${positionId}:\n\n${message}\n\nEnter number (1-5) or cancel:`);

    if (choice && !isNaN(choice)) {
        const idx = parseInt(choice) - 1;
        if (idx >= 0 && idx < top5.length) {
            state.assignToSlot(positionId, top5[idx].casteller.name);
            bus.emit(Events.UI_REFRESH);
            state.saveHistory(`Assigned ${top5[idx].casteller.name} to ${positionId}`);
        }
    }
}
