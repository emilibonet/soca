/**
 * Crossa overlap resolution using perpendicular bisector clipping.
 * Pure DOM geometry — no imports from other modules.
 */

export function resolveCrossaOverlaps() {
    const wrappers = Array.from(document.querySelectorAll('.crossa-wrapper'));

    // Reset all polygons to their original points
    wrappers.forEach(w => {
        const poly = w.querySelector('.crossa-rect polygon');
        if (poly && poly.dataset.originalPoints) {
            poly.setAttribute('points', poly.dataset.originalPoints);
        }
    });

    if (wrappers.length < 2) return;

    // Collect info: bounding rect in screen coords + SVG viewBox mapping
    const infos = wrappers.map(w => {
        const r = w.getBoundingClientRect();
        const svg = w.querySelector('.crossa-rect');
        const vb = svg.getAttribute('viewBox').split(/\s+/).map(Number);
        return {
            el: w,
            cx: r.left + r.width / 2,
            cy: r.top + r.height / 2,
            rect: r,
            vb: { x: vb[0], y: vb[1], w: vb[2], h: vb[3] },
            svg: svg
        };
    });

    // For each pair, check overlap and clip SVG polygons
    for (let i = 0; i < infos.length; i++) {
        for (let j = i + 1; j < infos.length; j++) {
            const a = infos[i], b = infos[j];
            // AABB overlap test
            if (a.rect.right <= b.rect.left || b.rect.right <= a.rect.left ||
                a.rect.bottom <= b.rect.top || b.rect.bottom <= a.rect.top) {
                continue;
            }

            // Perpendicular bisector in screen coords
            const mx = (a.cx + b.cx) / 2;
            const my = (a.cy + b.cy) / 2;
            const nx = b.cx - a.cx;
            const ny = b.cy - a.cy;

            clipCrossaPolygon(a, nx, ny, mx, my, -1);
            clipCrossaPolygon(b, nx, ny, mx, my, +1);
        }
    }
}

function screenToSVG(screenX, screenY, rect, vb) {
    return {
        x: vb.x + (screenX - rect.left) / rect.width * vb.w,
        y: vb.y + (screenY - rect.top) / rect.height * vb.h
    };
}

function svgToScreen(svgX, svgY, rect, vb) {
    return {
        x: rect.left + (svgX - vb.x) / vb.w * rect.width,
        y: rect.top + (svgY - vb.y) / vb.h * rect.height
    };
}

function clipCrossaPolygon(info, nx, ny, mx, my, sign) {
    const poly = info.svg.querySelector('polygon');
    if (!poly) return;

    // Parse current SVG polygon points
    const pointsStr = poly.getAttribute('points');
    const svgPoly = pointsStr.split(/\s+/).map(pair => {
        const [x, y] = pair.split(',').map(Number);
        return { x, y };
    }).filter(p => !isNaN(p.x) && !isNaN(p.y));

    if (svgPoly.length < 3) return;

    // Convert SVG polygon to screen coords, clip, convert back
    const screenPoly = svgPoly.map(p => svgToScreen(p.x, p.y, info.rect, info.vb));

    // Signed distance from bisector in screen space
    function dist(p) { return nx * (p.x - mx) + ny * (p.y - my); }
    function keeps(d) { return sign < 0 ? d <= 0.5 : d >= -0.5; }

    // Sutherland-Hodgman clip
    const clipped = [];
    for (let i = 0; i < screenPoly.length; i++) {
        const curr = screenPoly[i];
        const next = screenPoly[(i + 1) % screenPoly.length];
        const dCurr = dist(curr);
        const dNext = dist(next);
        const currIn = keeps(dCurr);
        const nextIn = keeps(dNext);

        if (currIn) clipped.push(curr);
        if (currIn !== nextIn) {
            const t = dCurr / (dCurr - dNext);
            clipped.push({
                x: curr.x + t * (next.x - curr.x),
                y: curr.y + t * (next.y - curr.y)
            });
        }
    }

    if (clipped.length < 3) return;

    // Check if nothing was actually clipped
    if (clipped.length === screenPoly.length && screenPoly.every(p => keeps(dist(p)))) {
        return;
    }

    // Convert back to SVG viewBox coordinates and update polygon
    const newPoints = clipped.map(p => {
        const sv = screenToSVG(p.x, p.y, info.rect, info.vb);
        return `${sv.x.toFixed(1)},${sv.y.toFixed(1)}`;
    }).join(' ');

    poly.setAttribute('points', newPoints);
}
