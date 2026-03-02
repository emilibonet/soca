/**
 * Application configuration constants.
 * 
 * DEFAULT_CONFIG:     Optimization parameters, position configs, queue configs.
 * MANS_GEOMETRY:      Per-column geometry for dynamically positioned mans SVG wrappers.
 * MATRIX_GEOMETRY:    Per-matrix geometry for SVG slot wrappers (laterals + daus).
 * MATRIX_CONFIGS:     Matrix configs: fused laterals + daus between each pair of columns.
 */

export const DEFAULT_CONFIG = {
    optimization: {
        method: 'adaptive_simulated_annealing',
        use_weight: false
    },
    positions: {
        baix: {
            height_ratio_min: 1.0,
            height_ratio_max: 1.0,
            height_weight: 0.0,
            expertise_weight: 1.0,
            weight_factor: 0.5,
            height_penalty_factor: 0.10,
            weight_preference: 'heavier'
        },
        crossa: {
            height_ratio_min: 0.90,
            height_ratio_max: 0.95,
            height_weight: 1.0,
            expertise_weight: 0.5,
            height_similarity_weight: 0.5,
            weight_factor: 0.2,
            height_penalty_factor: 0.10,
            weight_preference: 'neutral'
        },
        contrafort: {
            height_ratio_min: 1.00,
            height_ratio_max: 1.05,
            height_weight: 1.0,
            expertise_weight: 0.5,
            height_similarity_weight: 0.3,
            weight_factor: 0.3,
            height_penalty_factor: 0.10,
            weight_preference: 'heavier'
        },
        agulla: {
            height_ratio_min: 0.50,
            height_ratio_max: 0.515,
            height_weight: 1.0,
            expertise_weight: 0.5,
            weight_factor: 0.3,
            height_penalty_factor: 0.10,
            weight_preference: 'lighter'
        }
    },
    queues: {
        mans: {
            max_depth: 4,
            height_ratio_min: 0.50,
            height_ratio_max: 0.60,
            queue_height_ratio_min: 0.975,
            queue_height_ratio_max: 1.00,
            height_weight: 0.6,
            expertise_weight: 0.5,
            height_similarity_weight: 0.5,
            weight_factor: 0.2,
            height_penalty_factor: 1.0,
            weight_preference: 'heavier'
        },
        daus: {
            max_depth: 3,
            height_ratio_min: 0.50,
            height_ratio_max: 0.60,
            queue_height_ratio_min: 0.975,
            queue_height_ratio_max: 1.00,
            height_weight: 0.6,
            expertise_weight: 0.5,
            height_similarity_weight: 0.5,
            weight_factor: 0.3,
            height_penalty_factor: 1.0,
            weight_preference: 'heavier'
        },
        laterals: {
            max_depth: 2,
            height_ratio_min: 0.48,
            height_ratio_max: 0.50,
            queue_height_ratio_min: 0.975,
            queue_height_ratio_max: 1.0,
            height_weight: 0.8,
            expertise_weight: 0.3,
            height_similarity_weight: 0.5,
            weight_factor: 0.2,
            height_penalty_factor: 1.0,
            weight_preference: 'neutral'
        }
    }
};

/**
 * Per-column geometry for dynamically positioned mans SVG wrappers.
 * baseLeft / baseTop: offsets (px) from 50% / 45% for depth 0.
 * dLeft / dTop: per-depth increments (outward from centroid).
 */
export const MANS_GEOMETRY = {
    'Rengla': {
        baseLeft: -60, baseTop: 157, dLeft: 0, dTop: 55,
        w: 120, h: 55,
        viewBox: '0 0 120 55',
        points: '0,0 120,0 120,55 0,55',
        stroke: 'var(--rengla)',
        textX: 60, textY: 35
    },
    'Buida': {
        baseLeft: -212, baseTop: -159, dLeft: -47, dTop: -28,
        w: 108, h: 133,
        viewBox: '0 0 108 133',
        points: '48,132 108,27 60,0 0,104',
        stroke: 'var(--buida)',
        textX: 54, textY: 67
    },
    'Plena': {
        baseLeft: 104, baseTop: -159, dLeft: 47, dTop: -28,
        w: 108, h: 133,
        viewBox: '0 0 108 133',
        points: '60,132 0,27 48,0 108,104',
        stroke: 'var(--plena)',
        textX: 54, textY: 67
    }
};

/**
 * Per-matrix geometry for SVG slot wrappers (laterals + daus).
 * Each matrix sits in the gap between two adjacent columns.
 */
export const MATRIX_GEOMETRY = {
    'R\u2194P': {
        outX: 0.866, outY: 0.5,
        edgeX: -0.5, edgeY: 0.866,
        originX: 181, originY: 105,
        depthSpacing: 58,
        colSpacing: 125,
        w: 108, h: 132,
        viewBox: '0 0 108 132',
        points: '60,0 108,28 48,132 0,104',
        textX: 54, textY: 66,
        strokeColors: ['var(--rengla)', 'var(--col)', 'var(--plena)']
    },
    'P\u2194B': {
        outX: 0, outY: -1,
        edgeX: -1, edgeY: 0,
        originX: 0, originY: -210,
        depthSpacing: 58,
        colSpacing: 125,
        w: 120, h: 55,
        viewBox: '0 0 120 55',
        points: '0,0 120,0 120,55 0,55',
        textX: 60, textY: 30,
        strokeColors: ['var(--plena)', 'var(--col)', 'var(--buida)']
    },
    'B\u2194R': {
        outX: -0.866, outY: 0.5,
        edgeX: 0.5, edgeY: 0.866,
        originX: -181, originY: 105,
        depthSpacing: 58,
        colSpacing: 125,
        w: 108, h: 132,
        viewBox: '0 0 108 132',
        points: '48,0 108,104 60,132 0,28',
        textX: 54, textY: 66,
        strokeColors: ['var(--buida)', 'var(--col)', 'var(--rengla)']
    }
};

/**
 * Matrix configs: fused laterals + daus between each pair of columns.
 * Each matrix has 3 columns: lateral-left, daus, lateral-right.
 */
export const MATRIX_CONFIGS = [
    {
        id: 'R↔P',
        label: 'R ↔ P',
        columns: [
            { queueType: 'laterals', queueId: 'Rengla-right', header: 'Lat R→' },
            { queueType: 'daus', queueId: 'R↔P', header: 'Daus' },
            { queueType: 'laterals', queueId: 'Plena-left', header: '←Lat P' }
        ],
        cssClass: 'matrix-rp'
    },
    {
        id: 'P↔B',
        label: 'P ↔ B',
        columns: [
            { queueType: 'laterals', queueId: 'Plena-right', header: 'Lat P→' },
            { queueType: 'daus', queueId: 'P↔B', header: 'Daus' },
            { queueType: 'laterals', queueId: 'Buida-left', header: '←Lat B' }
        ],
        cssClass: 'matrix-pb'
    },
    {
        id: 'B↔R',
        label: 'B ↔ R',
        columns: [
            { queueType: 'laterals', queueId: 'Buida-right', header: 'Lat B→' },
            { queueType: 'daus', queueId: 'B↔R', header: 'Daus' },
            { queueType: 'laterals', queueId: 'Rengla-left', header: '←Lat R' }
        ],
        cssClass: 'matrix-br'
    }
];

/** The three pinya column names */
export const COLUMNS = ['Rengla', 'Plena', 'Buida'];
