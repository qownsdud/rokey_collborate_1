"""Phase 3: Vector path extraction from B&W line art images."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


@dataclass
class ExtractionResult:
    """Result of vector path extraction from a raster line art image."""

    width: int
    height: int
    strokes: list[list[tuple[float, float]]]
    raw_point_count: int
    simplified_point_count: int
    draw_length_px: float
    travel_length_px: float
    elapsed_sec: float
    binary_inv: np.ndarray
    skeleton: np.ndarray


def _binarize_lineart(gray: np.ndarray, threshold: int | None = None, morph_clean: bool = False) -> np.ndarray:
    """Convert grayscale line art (black lines on white bg) to binary foreground (255=line, 0=bg)."""
    if threshold is None:
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary_inv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    if morph_clean:
        kernel = np.ones((2, 2), np.uint8)
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    return binary_inv


def _trace_skeleton_strokes(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace 1-pixel-wide skeleton image into continuous (x, y) polylines without dropping junction edges."""
    skel_bin = (skeleton > 0).astype(np.uint8)
    h, w = skel_bin.shape

    ys, xs = np.where(skel_bin == 1)
    nodes = set(zip(xs.tolist(), ys.tolist()))
    if not nodes:
        return []

    dirs = ((0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (1, -1), (1, 1), (-1, 1))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in nodes:
        nbrs: list[tuple[int, int]] = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skel_bin[ny, nx]:
                nbrs.append((nx, ny))
        adj[(x, y)] = nbrs

    used_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def make_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    def walk_edge(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        chain = [start, nxt]
        used_edges.add(make_edge(start, nxt))
        prev, cur = start, nxt
        while len(adj[cur]) == 2:
            n0, n1 = adj[cur]
            cand = n1 if n0 == prev else n0
            e = make_edge(cur, cand)
            if e in used_edges:
                break
            used_edges.add(e)
            chain.append(cand)
            prev, cur = cur, cand
            if cur == start:
                break
        return chain

    strokes: list[list[tuple[int, int]]] = []

    non_regular = sorted(
        (n for n, nbrs in adj.items() if len(nbrs) != 2),
        key=lambda n: (len(adj[n]), n[1], n[0]),
    )
    for node in non_regular:
        for nbr in adj[node]:
            if make_edge(node, nbr) not in used_edges:
                strokes.append(walk_edge(node, nbr))

    for node in sorted(nodes, key=lambda n: (n[1], n[0])):
        for nbr in adj[node]:
            if make_edge(node, nbr) not in used_edges:
                strokes.append(walk_edge(node, nbr))

    return strokes


def _extract_thick_region_contours(binary_inv: np.ndarray, min_half_width: float = 1.8) -> list[list[tuple[int, int]]]:
    """Extract contours of thick/filled regions (eyes, lips, bold hair strokes) to preserve line weight."""
    dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)
    thick_seed = (dist >= min_half_width).astype(np.uint8) * 255
    if not np.any(thick_seed):
        return []

    # Expand seeds slightly to recover the boundary of thick strokes only
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick_mask = cv2.dilate(thick_seed, kernel, iterations=1)
    thick_region = cv2.bitwise_and(binary_inv, thick_mask)

    contours, _ = cv2.findContours(thick_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out: list[list[tuple[int, int]]] = []
    for cnt in contours:
        if len(cnt) < 6:
            continue
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        pts.append(pts[0])
        out.append(pts)
    return out


def _polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return total


def _bridge_close_strokes(
    strokes: list[list[tuple[float, float]]],
    max_gap_px: float = 4.5,
) -> list[list[tuple[float, float]]]:
    """Merge open strokes whose endpoints are within max_gap_px to reduce fragmented gaps."""
    if max_gap_px <= 0 or len(strokes) < 2:
        return strokes

    pool = [list(s) for s in strokes]
    merged: list[list[tuple[float, float]]] = []

    while pool:
        cur = pool.pop(0)
        # Do not merge closed loops
        if len(cur) >= 3 and math.hypot(cur[0][0] - cur[-1][0], cur[0][1] - cur[-1][1]) < 1.5:
            merged.append(cur)
            continue

        changed = True
        while changed and pool:
            changed = False
            best_i = -1
            best_dist = max_gap_px
            mode = ""  # end->start, end->end, start->start, start->end

            cx0, cy0 = cur[0]
            cx1, cy1 = cur[-1]

            for i, cand in enumerate(pool):
                if len(cand) >= 3 and math.hypot(cand[0][0] - cand[-1][0], cand[0][1] - cand[-1][1]) < 1.5:
                    continue
                ax0, ay0 = cand[0]
                ax1, ay1 = cand[-1]

                checks = (
                    (math.hypot(cx1 - ax0, cy1 - ay0), "e_s"),
                    (math.hypot(cx1 - ax1, cy1 - ay1), "e_e"),
                    (math.hypot(cx0 - ax1, cy0 - ay1), "s_e"),
                    (math.hypot(cx0 - ax0, cy0 - ay0), "s_s"),
                )
                for d, m in checks:
                    if d < best_dist:
                        best_dist = d
                        best_i = i
                        mode = m

            if best_i >= 0:
                nxt = pool.pop(best_i)
                if mode == "e_s":
                    cur = cur + nxt
                elif mode == "e_e":
                    cur = cur + list(reversed(nxt))
                elif mode == "s_e":
                    cur = nxt + cur
                elif mode == "s_s":
                    cur = list(reversed(nxt)) + cur
                changed = True

        merged.append(cur)

    return merged


def _chaikin_smooth(points: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
    """Smooth jagged pixel staircases using Chaikin's corner-cutting algorithm."""
    if iterations <= 0 or len(points) < 3:
        return points

    is_closed = math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 1.5
    pts = list(points[:-1]) if is_closed else list(points)

    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts: list[tuple[float, float]] = []
        if not is_closed:
            new_pts.append(pts[0])
        n = len(pts)
        limit = n if is_closed else n - 1
        for i in range(limit):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.extend((q, r))
        if not is_closed:
            new_pts.append(pts[-1])
        pts = new_pts

    if is_closed and pts:
        pts.append(pts[0])
    return pts


def _simplify_and_filter(
    raw_strokes: list[list[tuple[int, int]]],
    epsilon: float,
    min_length_px: float,
    min_points: int,
    bridge_gap_px: float = 4.5,
    smooth_iterations: int = 2,
) -> tuple[list[list[tuple[float, float]]], int, int]:
    """Bridge small gaps, smooth staircases, simplify with Douglas-Peucker, and filter noise."""
    raw_count = sum(len(s) for s in raw_strokes)
    float_strokes = [[(float(x), float(y)) for x, y in s] for s in raw_strokes if len(s) >= 2]
    bridged = _bridge_close_strokes(float_strokes, max_gap_px=bridge_gap_px)

    filtered: list[list[tuple[float, float]]] = []
    for stroke in bridged:
        if _polyline_length(stroke) < min_length_px:
            continue
        smoothed = _chaikin_smooth(stroke, iterations=smooth_iterations)
        is_closed = len(smoothed) >= 3 and math.hypot(smoothed[0][0] - smoothed[-1][0], smoothed[0][1] - smoothed[-1][1]) < 1.0
        arr = np.array(smoothed[:-1] if is_closed else smoothed, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(arr, epsilon=epsilon, closed=is_closed)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if is_closed and len(pts) >= 2:
            pts.append(pts[0])

        if len(pts) < min_points or _polyline_length(pts) < min_length_px:
            continue
        filtered.append(pts)

    simplified_count = sum(len(s) for s in filtered)
    return filtered, raw_count, simplified_count


def _order_strokes_tsp(strokes: list[list[tuple[float, float]]]) -> tuple[list[list[tuple[float, float]]], float]:
    """Order strokes using nearest-neighbor TSP heuristic (checking both start and end)."""
    if not strokes:
        return [], 0.0

    remaining = list(range(len(strokes)))
    best_idx = min(
        remaining,
        key=lambda idx: min(
            math.hypot(strokes[idx][0][0], strokes[idx][0][1]),
            math.hypot(strokes[idx][-1][0], strokes[idx][-1][1]),
        ),
    )
    remaining.remove(best_idx)
    first = strokes[best_idx]
    if math.hypot(first[-1][0], first[-1][1]) < math.hypot(first[0][0], first[0][1]):
        first = list(reversed(first))

    ordered = [first]
    cur_x, cur_y = first[-1]
    travel_dist = 0.0

    while remaining:
        best_pos = 0
        best_dist = float("inf")
        reverse_flag = False

        for pos, idx in enumerate(remaining):
            s = strokes[idx]
            d_start = math.hypot(s[0][0] - cur_x, s[0][1] - cur_y)
            d_end = math.hypot(s[-1][0] - cur_x, s[-1][1] - cur_y)
            if d_start < best_dist:
                best_dist = d_start
                best_pos = pos
                reverse_flag = False
            if d_end < best_dist:
                best_dist = d_end
                best_pos = pos
                reverse_flag = True

        chosen_idx = remaining.pop(best_pos)
        chosen = list(reversed(strokes[chosen_idx])) if reverse_flag else strokes[chosen_idx]
        ordered.append(chosen)
        travel_dist += best_dist
        cur_x, cur_y = chosen[-1]

    return ordered, travel_dist


def extract_vector_paths(
    image_path: str,
    epsilon: float = 0.45,
    min_length_px: float = 2.5,
    min_points: int = 2,
    threshold: int | None = None,
    morph_clean: bool = False,
    preserve_thick_contours: bool = True,
    bridge_gap_px: float = 4.5,
    smooth_iterations: int = 2,
) -> ExtractionResult:
    """Extract ordered, smooth vector strokes preserving thin lines and thick feature contours."""
    t0 = time.perf_counter()
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    h, w = gray.shape
    binary_inv = _binarize_lineart(gray, threshold=threshold, morph_clean=morph_clean)
    skeleton = cv2.ximgproc.thinning(binary_inv, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    raw_strokes = _trace_skeleton_strokes(skeleton)
    if preserve_thick_contours:
        thick_contours = _extract_thick_region_contours(binary_inv, min_half_width=1.8)
        raw_strokes.extend(thick_contours)

    simplified, raw_cnt, simp_cnt = _simplify_and_filter(
        raw_strokes,
        epsilon=epsilon,
        min_length_px=min_length_px,
        min_points=min_points,
        bridge_gap_px=bridge_gap_px,
        smooth_iterations=smooth_iterations,
    )
    ordered, travel_px = _order_strokes_tsp(simplified)
    draw_px = sum(_polyline_length(s) for s in ordered)
    elapsed = time.perf_counter() - t0

    return ExtractionResult(
        width=w,
        height=h,
        strokes=ordered,
        raw_point_count=raw_cnt,
        simplified_point_count=simp_cnt,
        draw_length_px=draw_px,
        travel_length_px=travel_px,
        elapsed_sec=elapsed,
        binary_inv=binary_inv,
        skeleton=skeleton,
    )
