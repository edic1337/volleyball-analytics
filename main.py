"""
main.py — Четырехпроходный физический пайплайн поверх детекций YOLO:
  Проход 1: Look-Ahead 2D-трекинг (захват вбросов с conf >= 0.24 по 3-кадровой коллинеарности).
  Проход 2: Гравитационная фильтрация фаз 'мяч в руках игрока' (0.0 км/ч) и первичная 3D-сборка.
  Проход 3: Обратная проекция 3D-физики на кэш YOLO и уточнение кадров удара.
  Проход 4: Финальная 3D-реконструкция с 6D Bundle Adjustment.
"""
import os
import pickle
import cv2
import numpy as np
from ultralytics import YOLO

from ball_speed import (
    RADIUS, project, triangulate_from_depth, ray_court_depth_bounds,
    court_depth_stats, calibrate_box_scale, solve_connected_rally_arcs,
)

VIDEO_PATH = "trimmed_clip.mp4"
YOLO_WEIGHTS = "runs/detect/train-18/weights/best_openvino_model/"
CALIB_PATH = "camera_calib_opti.npz"
OUTPUT_VIDEO = "trimmed_clip_ekf_long.mp4"
OUTPUT_TRAJECTORY = "ball_trajectory_trimmed_clip_ekf_long.npy"

CONF_THRESHOLD = 0.22
IOU_THRESHOLD = 0.45
IMG_SIZE = 1280
SHOW_LIVE_PREVIEW = True


def load_calib(path):
    calib = np.load(path)
    return calib["K"], calib["R"], calib["T"]


def extract_or_load_yolo_detections(video_path, model_path):
    cache_path = f"{video_path}.yolo_cache.pkl"
    if os.path.exists(cache_path):
        print(f"Загружаю кэш детекций YOLO из {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Первичный прогон YOLO по видео (результат будет сохранен в кэш)...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_cands = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(
            frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE, verbose=False
        )[0]
        cands = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                bsize = min(x2 - x1, y2 - y1)
                cands.append((
                    float(cx), float(cy), float(bsize), float(c),
                    (int(x1), int(y1), int(x2), int(y2))
                ))
        all_cands.append(cands)
        idx += 1
        if idx % 200 == 0:
            print(f"  YOLO обработал кадров: {idx}")

    cap.release()
    data = dict(fps=fps, all_cands=all_cands)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Кэш детекций сохранен: {cache_path} ({len(all_cands)} кадров)")
    return data


def find_static_background_balls(all_cands, grid_px=14.0, min_persistence_ratio=0.04):
    counts = {}
    n_frames = max(len(all_cands), 1)
    for cands in all_cands:
        seen_cells = set()
        for c in cands:
            cell = (int(round(c[0] / grid_px)), int(round(c[1] / grid_px)))
            seen_cells.add(cell)
        for cell in seen_cells:
            counts[cell] = counts.get(cell, 0) + 1

    static_centers = []
    for (gx, gy), cnt in counts.items():
        if cnt > max(32, int(n_frames * min_persistence_ratio)):
            static_centers.append((gx * grid_px, gy * grid_px))
    return static_centers


def is_static_false_positive(u, v, static_centers, radius=22.0):
    for sx, sy in static_centers:
        if np.hypot(u - sx, v - sy) < radius:
            return True
    return False


def is_valid_ball_candidate(cand, K, R, T, is_tracking=False):
    cx, cy, bsize, conf, (x1, y1, x2, y2) = cand
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = max(w, h) / float(min(w, h))

    if aspect > 2.15 or (aspect > 1.75 and conf < 0.42):
        return False
    if bsize < 4.5 or bsize > 68.0:
        return False

    if conf < 0.45:
        f_eff = 0.5 * (K[0, 0] + K[1, 1])
        z_raw = f_eff * (2.0 * RADIUS) / bsize
        z_lo, z_hi = ray_court_depth_bounds((cx, cy), K, R, T)
        if not (z_lo * 0.45 <= z_raw <= z_hi * 1.65):
            return False

    return True


def _has_kinematic_continuation(all_cands, frame_idx, cand, static_balls, K, R, T, require_two_steps=False):
    """
    Проверяет на 1-2 кадра вперед наличие плавного продолжения движения.
    Если require_two_steps=True (при низком conf < 0.40 на вбросе мяча), требует 3 коллинеарные точки!
    """
    u0, v0, b0, _, _ = cand
    n = len(all_cands)

    for step1 in (1, 2):
        if frame_idx + step1 >= n:
            continue
        for c1 in all_cands[frame_idx + step1]:
            if is_static_false_positive(c1[0], c1[1], static_balls):
                continue
            if not is_valid_ball_candidate(c1, K, R, T):
                continue
            du1 = c1[0] - u0
            dv1 = c1[1] - v0
            dist1 = np.hypot(du1, dv1)
            if dist1 > 115.0 * step1 or abs(c1[2] - b0) > max(12.0, 0.55 * b0):
                continue

            if not require_two_steps:
                return True

            # Для слабого conf проверяем второй шаг вдоль того же вектора скорости
            u_exp = c1[0] + du1 / step1
            v_exp = c1[1] + dv1 / step1 + 8.0
            if frame_idx + step1 + 1 < n:
                for c2 in all_cands[frame_idx + step1 + 1]:
                    if is_static_false_positive(c2[0], c2[1], static_balls):
                        continue
                    if np.hypot(c2[0] - u_exp, c2[1] - v_exp) < 42.0:
                        return True
    return False


def _clean_2d_track_outliers(chosen_list, all_cands, static_balls, K, R, T):
    n = len(chosen_list)
    cleaned = list(chosen_list)

    for i in range(2, n - 2):
        if cleaned[i] is None:
            continue
        neigh = [(k, cleaned[k]) for k in (i - 2, i - 1, i + 1, i + 2) if cleaned[k] is not None]
        if len(neigh) < 3:
            continue

        ks = np.array([k for k, _ in neigh], dtype=float)
        us = np.array([c[0] for _, c in neigh], dtype=float)
        vs = np.array([c[1] for _, c in neigh], dtype=float)
        deg = 2 if len(neigh) == 4 else 1
        u_interp = float(np.polyval(np.polyfit(ks - i, us, deg), 0.0))
        v_interp = float(np.polyval(np.polyfit(ks - i, vs, deg), 0.0))

        curr_u, curr_v, _, curr_conf, _ = cleaned[i]
        err_px = np.hypot(curr_u - u_interp, curr_v - v_interp)

        if err_px > 28.0 and (curr_conf < 0.55 or err_px > 48.0):
            best_rep = None
            best_d = 24.0
            for c in all_cands[i]:
                if is_static_false_positive(c[0], c[1], static_balls):
                    continue
                d = np.hypot(c[0] - u_interp, c[1] - v_interp)
                if d < best_d:
                    best_d, best_rep = d, c
            cleaned[i] = best_rep

    return cleaned


def track_2d_video(all_cands, fps, width, height, K, R, T):
    dt = 1.0 / fps
    static_balls = find_static_background_balls(all_cands)
    if static_balls:
        print(f"Обнаружено и замаскировано неподвижных фоновых мячей: {len(static_balls)}")

    n = len(all_cands)
    chosen_list = [None] * n
    out_of_fov_list = [False] * n

    history = []
    lost_in_fov = 0
    above_frame_mode = False
    above_poly_u = None
    above_poly_v = None
    t_exit = 0.0

    for i in range(n):
        t = i * dt
        raw_cands = [
            c for c in all_cands[i]
            if not is_static_false_positive(c[0], c[1], static_balls)
            and is_valid_ball_candidate(c, K, R, T, is_tracking=bool(history))
        ]

        if above_frame_mode:
            dt_out = t - t_exit
            u_pred = above_poly_u[0] + above_poly_u[1] * dt_out
            v_pred = above_poly_v[0] + above_poly_v[1] * dt_out + above_poly_v[2] * (dt_out ** 2)

            if dt_out > 1.85 or v_pred > height * 0.42:
                above_frame_mode = False
                history = []
            else:
                out_of_fov_list[i] = True
                best, best_d = None, 190.0
                for c in raw_cands:
                    if c[1] < 220.0:
                        d = np.hypot(c[0] - u_pred, c[1] - max(v_pred, 15.0))
                        if d < best_d:
                            best_d, best = d, c
                if best is not None:
                    chosen_list[i] = best
                    out_of_fov_list[i] = False
                    above_frame_mode = False
                    history.append((t, best[0], best[1], best[2], best[3]))
                    lost_in_fov = 0
                continue

        # Старт трека: разрешаем conf >= 0.40 (1 шаг) ИЛИ conf >= 0.24 при 3-кадровой коллинеарности (вброс мяча!)
        if not history:
            valid_starts = [
                c for c in raw_cands
                if (c[3] >= 0.40 and _has_kinematic_continuation(all_cands, i, c, static_balls, K, R, T, False))
                or (c[3] >= 0.24 and _has_kinematic_continuation(all_cands, i, c, static_balls, K, R, T, True))
            ]
            if valid_starts:
                best = max(valid_starts, key=lambda c: c[3])
                chosen_list[i] = best
                history = [(t, best[0], best[1], best[2], best[3])]
                lost_in_fov = 0
            continue

        t_last, u_last, v_last, b_last, _ = history[-1]
        dt_last = max(t - t_last, dt)

        if len(history) >= 3:
            pts_h = np.array(history[-min(len(history), 5):], dtype=float)
            ts_h = pts_h[:, 0] - t_last
            deg_h = 2 if len(pts_h) >= 4 else 1
            pu = np.polyfit(ts_h, pts_h[:, 1], 1)
            pv = np.polyfit(ts_h, pts_h[:, 2], deg_h)
            u_pred = float(np.polyval(pu, dt_last))
            v_pred = float(np.polyval(pv, dt_last))
            vu = float(pu[0])
            vv = float(pv[-2])
        elif len(history) == 2:
            t_prev, u_prev, v_prev, _, _ = history[-2]
            dt_prev = max(t_last - t_prev, dt)
            vu = (u_last - u_prev) / dt_prev
            vv = (v_last - v_prev) / dt_prev
            u_pred = u_last + vu * dt_last
            v_pred = v_last + vv * dt_last + 0.5 * 650.0 * (dt_last ** 2)
        else:
            vu, vv = 0.0, 0.0
            u_pred, v_pred = u_last, v_last

        if v_pred < 12.0 and v_last < 98.0 and vv < -75.0 and len(history) >= 4:
            top_cand = None
            for c in raw_cands:
                if c[1] < 105.0 and np.hypot(c[0] - u_pred, c[1] - max(v_pred, 0.0)) < 85.0:
                    top_cand = c
                    break
            if top_cand is None:
                pts = np.array(history[-min(len(history), 8):], dtype=float)
                ts_rel = pts[:, 0] - pts[-1, 0]
                poly_u = np.polyfit(ts_rel, pts[:, 1], 1)
                poly_v = np.polyfit(ts_rel, pts[:, 2], 2)
                a_v = max(float(poly_v[0]), 250.0)
                above_poly_u = [float(u_last), float(poly_u[0])]
                above_poly_v = [float(v_last), float(poly_v[1]), a_v]
                t_exit = t_last
                above_frame_mode = True
                out_of_fov_list[i] = True
                continue

        best = None
        best_score = float("inf")
        max_gate = 120.0 + 25.0 * min(lost_in_fov, 3)

        for c in raw_cands:
            d_pred = np.hypot(c[0] - u_pred, c[1] - v_pred)
            d_last = np.hypot(c[0] - u_last, c[1] - v_last)
            d_eff = d_pred if len(history) >= 3 else min(d_pred, d_last * 1.1)

            if d_eff > max_gate:
                continue
            if c[3] < 0.36 and d_pred > 48.0:
                continue

            box_jump = abs(c[2] - b_last) / max(b_last, 8.0)
            if c[3] < 0.40 and box_jump > 0.65:
                continue

            la_bonus = 0.78 if _has_kinematic_continuation(all_cands, i, c, static_balls, K, R, T) else 1.15
            score = (d_eff * (1.0 + 0.35 * box_jump) * la_bonus) / (max(c[3], 0.25) ** 0.7)

            if score < best_score:
                best_score, best = score, c

        if best is not None:
            chosen_list[i] = best
            history.append((t, best[0], best[1], best[2], best[3]))
            if len(history) > 25:
                history.pop(0)
            lost_in_fov = 0
        else:
            lost_in_fov += 1
            if lost_in_fov > 6:
                history = []
                lost_in_fov = 0

    chosen_list = _clean_2d_track_outliers(chosen_list, all_cands, static_balls, K, R, T)
    return chosen_list, out_of_fov_list, static_balls


def _is_free_flight_arc(arc_arr, K=None):
    """
    Проверяет, действительно ли мяч летит по воздуху под действием гравитации,
    сверяя экранное ускорение d2v/dt2 с размером рамки YOLO (защита от мяча в руках).
    """
    if len(arc_arr) < 5:
        return False

    times = arc_arr[:, 0]
    u = arc_arr[:, 1]
    v = arc_arr[:, 2]
    boxes = arc_arr[:, 3]

    span_2d = float(np.hypot(u.max() - u.min(), v.max() - v.min()))
    step_dt = np.maximum(np.diff(times), 1e-3)
    step_spd = np.hypot(np.diff(u), np.diff(v)) / step_dt
    med_spd_2d = float(np.median(step_spd))

    # 1. Медленное движение или малый размах — мяч в руках игрока
    if span_2d < 70.0 or med_spd_2d < 185.0:
        return False

    # 2. Сравнение вертикального ускорения на экране с ожидаемой гравитацией для данного размера бокса
    tau = times - 0.5 * (times[0] + times[-1])
    poly_v = np.polyfit(tau, v, 2)
    a_v = 2.0 * float(poly_v[0])

    f_eff = 0.5 * (K[0, 0] + K[1, 1]) if K is not None else 1150.0
    med_box = float(np.median(boxes[boxes > 4.0])) if np.any(boxes > 4.0) else 18.0
    z_est_box = f_eff * (2.0 * RADIUS) / max(med_box, 6.0)
    # Ожидаемое экранное ускорение гравитации на глубине z_est_box
    a_v_expected = (f_eff * 9.80665 * 0.85) / np.clip(z_est_box, 5.5, 32.0)

    # Если экранное ускорение меньше 38% от гравитации на этой глубине и нет огромной скорости удара,
    # значит мяч поддерживают руки игрока!
    if a_v < 0.38 * a_v_expected and (med_spd_2d < 440.0 or span_2d < 180.0):
        return False

    return True


def segment_rallies_and_arcs(chosen_list, out_of_fov_list, fps):
    dt = 1.0 / fps
    rallies = []
    curr_rally = []
    last_frame = None

    for i, ch in enumerate(chosen_list):
        if ch is not None:
            if last_frame is not None:
                gap = i - last_frame
                was_above_fov = any(out_of_fov_list[last_frame + 1:i]) and ch[1] < 220.0
                if (gap > 6 and not was_above_fov) or gap > 42:
                    if len(curr_rally) >= 6:
                        rallies.append(curr_rally)
                    curr_rally = []
            curr_rally.append([i * dt, ch[0], ch[1], ch[2], ch[3], i])
            last_frame = i

    if len(curr_rally) >= 6:
        rallies.append(curr_rally)

    stationary_frames = {}
    all_rally_arcs = []

    for rally in rallies:
        arr = np.asarray(rally, dtype=float)
        n = len(arr)

        # 1. Отделяем фазы стояния/переноса мяча в руках игрока
        is_stat = np.zeros(n, dtype=bool)
        for k in range(n):
            w_lo = max(0, k - 3)
            w_hi = min(n, k + 4)
            win = arr[w_lo:w_hi]
            span = np.hypot(win[:, 1].max() - win[:, 1].min(), win[:, 2].max() - win[:, 2].min())
            dt_win = max(float(win[-1, 0] - win[0, 0]), 1e-3)
            avg_spd = span / dt_win

            # Мяч в руках: либо малый размах (< 24 px), либо медленный перенос (< 130 px/s) без падения
            if span < 24.0 or avg_spd < 125.0:
                is_stat[k] = True
                stationary_frames[int(arr[k, 5])] = (arr[k, 1], arr[k, 2], arr[k, 3])

        active_chunks = []
        curr_chunk = []
        for k in range(n):
            if not is_stat[k]:
                if curr_chunk and (arr[k, 5] - curr_chunk[-1][5]) > 6:
                    f_prev = int(curr_chunk[-1][5])
                    f_curr = int(arr[k, 5])
                    if not any(out_of_fov_list[f_prev + 1:f_curr]):
                        if len(curr_chunk) >= 6:
                            active_chunks.append(curr_chunk)
                        curr_chunk = []
                curr_chunk.append(rally[k])
            else:
                if len(curr_chunk) >= 6:
                    active_chunks.append(curr_chunk)
                curr_chunk = []
        if len(curr_chunk) >= 6:
            active_chunks.append(curr_chunk)

        for chunk in active_chunks:
            act = np.asarray(chunk, dtype=float)
            m = len(act)
            if m < 6:
                for row in act:
                    stationary_frames[int(row[5])] = (row[1], row[2], row[3])
                continue

            impulse_scores = np.zeros(m, dtype=float)
            W = 3
            G_PX_EST = 750.0

            for k in range(W, m - W):
                dt_in = float(act[k, 0] - act[k - W, 0])
                dt_out = float(act[k + W, 0] - act[k, 0])
                if dt_in > 0.30 or dt_out > 0.30 or dt_in < 1e-3 or dt_out < 1e-3:
                    continue

                v_in = (act[k, 1:3] - act[k - W, 1:3]) / dt_in
                v_out = (act[k + W, 1:3] - act[k, 1:3]) / dt_out

                dt_mid = 0.5 * (dt_in + dt_out)
                dv_u = float(v_out[0] - v_in[0])
                dv_v_raw = float(v_out[1] - v_in[1])
                dv_v_nongrav = dv_v_raw - G_PX_EST * dt_mid

                spd_out = float(np.hypot(v_out[0], v_out[1]))

                upward_kick = max(0.0, -dv_v_raw - 210.0) if spd_out > 125.0 else 0.0
                horiz_kick = max(0.0, abs(dv_u) - 230.0) if spd_out > 150.0 else 0.0
                spike_kick = max(0.0, dv_v_nongrav - 270.0) if spd_out > 360.0 else 0.0

                score = max(upward_kick, horiz_kick, spike_kick)
                if score > 0.0:
                    impulse_scores[k] = float(np.hypot(dv_u, dv_v_nongrav))

            hit_indices = [0]
            MIN_ARC_LEN = 7
            for k in range(W, m - W):
                if impulse_scores[k] <= 0.0:
                    continue
                lo = max(W, k - 4)
                hi = min(m - W, k + 5)
                if impulse_scores[k] == np.max(impulse_scores[lo:hi]):
                    if k - hit_indices[-1] >= MIN_ARC_LEN and (m - 1 - k) >= 5:
                        hit_indices.append(k)

            hit_indices.append(m - 1)

            for idx_pos in range(1, len(hit_indices) - 1):
                s_prev = hit_indices[idx_pos - 1]
                k_curr = hit_indices[idx_pos]
                e_next = hit_indices[idx_pos + 1]
                best_k = k_curr
                best_err = float("inf")
                for cand_k in range(max(s_prev + 4, k_curr - 2), min(e_next - 4, k_curr + 3)):
                    seg1 = act[s_prev:cand_k + 1]
                    seg2 = act[cand_k:e_next + 1]
                    t1 = seg1[:, 0] - seg1[0, 0]
                    t2 = seg2[:, 0] - seg2[0, 0]
                    err1 = np.sum(np.abs(np.polyval(np.polyfit(t1, seg1[:, 2], min(2, len(t1) - 1)), t1) - seg1[:, 2]))
                    err2 = np.sum(np.abs(np.polyval(np.polyfit(t2, seg2[:, 2], min(2, len(t2) - 1)), t2) - seg2[:, 2]))
                    if err1 + err2 < best_err:
                        best_err = err1 + err2
                        best_k = cand_k
                hit_indices[idx_pos] = best_k

            # Разбиваем на непрерывные цепочки ТОЛЬКО тех дуг, которые прошли тест свободного полета
            curr_valid_arcs = []
            for idx_i in range(len(hit_indices) - 1):
                s_i = hit_indices[idx_i]
                e_i = hit_indices[idx_i + 1]
                sub_arc = act[s_i:e_i + 1]
                if len(sub_arc) >= 5 and _is_free_flight_arc(sub_arc):
                    curr_valid_arcs.append(sub_arc)
                else:
                    # Небаллистический сегмент (мяч в руках у игрока!) -> помечаем как 0.0 km/h
                    for row in sub_arc:
                        stationary_frames[int(row[5])] = (row[1], row[2], row[3])
                    if curr_valid_arcs:
                        all_rally_arcs.append(curr_valid_arcs)
                        curr_valid_arcs = []

            if curr_valid_arcs:
                all_rally_arcs.append(curr_valid_arcs)

    return all_rally_arcs, stationary_frames


def refine_detections_with_3d_physics(chosen_list, frame_3d_map, all_cands, static_balls, K, R, T):
    refined = list(chosen_list)
    recovered_cnt = 0
    pruned_cnt = 0

    for f_idx, info3d in frame_3d_map.items():
        if f_idx < 0 or f_idx >= len(refined):
            continue
        state = info3d["state"]
        if float(info3d["speed"]) < 0.1:
            continue
        (u_phys, v_phys), _ = project(state, K, R, T)
        if not (np.isfinite(u_phys) and np.isfinite(v_phys)):
            continue

        curr = refined[f_idx]
        if curr is not None:
            dist_curr = np.hypot(curr[0] - u_phys, curr[1] - v_phys)
            if dist_curr <= 24.0:
                continue
            pruned_cnt += 1
            refined[f_idx] = None

        best_c = None
        best_d = 26.0
        for c in all_cands[f_idx]:
            if is_static_false_positive(c[0], c[1], static_balls):
                continue
            if not is_valid_ball_candidate(c, K, R, T, is_tracking=True):
                continue
            d = np.hypot(c[0] - u_phys, c[1] - v_phys)
            if d < best_d:
                best_d, best_c = d, c

        if best_c is not None:
            refined[f_idx] = best_c
            recovered_cnt += 1

    print(f"  Физическая фильтрация: подтянуто пропущенных/точных боксов: {recovered_cnt}, "
          f"удалено ложных выбросов: {pruned_cnt}")
    return refined


def draw_frame(frame, chosen, state, speed, v0_kmh, K, R, T, hit=False, out_of_fov=False):
    h, w = frame.shape[:2]
    if state is not None:
        (u_phys, v_phys), _ = project(state, K, R, T)
        color_phys = (255, 140, 0) if chosen is None else (0, 200, 255)
        if np.isfinite(u_phys) and np.isfinite(v_phys):
            u_i = int(np.clip(u_phys, 15, w - 240))
            v_i = int(np.clip(v_phys, 35, h - 20))
            if 0 <= u_phys < w and 0 <= v_phys < h:
                cv2.circle(frame, (int(round(u_phys)), int(round(v_phys))), 5, color_phys, -1)
            label = f"{speed * 3.6:.1f} km/h"
            if v0_kmh > 1.0:
                label += f" (v0: {v0_kmh:.0f})"
            cv2.putText(frame, label, (u_i + 12, v_i - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    if chosen is not None:
        cx, cy, box_size, conf, (x1, y1, x2, y2) = chosen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"YOLO {conf:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif out_of_fov:
        cv2.putText(frame, "BALL ABOVE FRAME (3D parabolic arc)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 180, 0), 2)

    if hit:
        cv2.putText(frame, "HIT!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def main():
    K, R, T = load_calib(CALIB_PATH)
    z_cmin, z_ccenter, z_cmax = court_depth_stats(K, R, T)
    print(f"Геометрия площадки по калибровке: глубина Z_cam от {z_cmin:.1f} м "
          f"до {z_cmax:.1f} м (центр корта: {z_ccenter:.1f} м)")

    cache = extract_or_load_yolo_detections(VIDEO_PATH, YOLO_WEIGHTS)
    fps = cache["fps"]
    all_cands = cache["all_cands"]

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("\n[Проход 1/4] Look-Ahead 2D-трекинг и фильтрация ложных боксов...")
    chosen_list, out_of_fov_list, static_balls = track_2d_video(all_cands, fps, width, height, K, R, T)
    all_rally_arcs, stationary_frames = segment_rallies_and_arcs(chosen_list, out_of_fov_list, fps)
    box_scale = calibrate_box_scale(all_rally_arcs, K, R, T)
    print(f"  Первичная калибровка масштаба боксов YOLO: x{box_scale:.2f}")

    print("\n[Проход 2/4] Первичная 3D-реконструкция для физической верификации трека...")
    pass1_3d_map = {}
    for arcs in all_rally_arcs:
        res_map, _ = solve_connected_rally_arcs(arcs, K, R, T, box_scale=box_scale, refine_6d=False)
        pass1_3d_map.update(res_map)

    print("\n[Проход 3/4] Обратный проход 3D-физикой по кэшу YOLO и уточнение ударов...")
    chosen_list = refine_detections_with_3d_physics(
        chosen_list, pass1_3d_map, all_cands, static_balls, K, R, T
    )
    all_rally_arcs, stationary_frames = segment_rallies_and_arcs(chosen_list, out_of_fov_list, fps)
    total_arcs = sum(len(r) for r in all_rally_arcs)
    box_scale = calibrate_box_scale(all_rally_arcs, K, R, T)
    print(f"  После уточнения: эпизодов = {len(all_rally_arcs)}, чистых дуг = {total_arcs}, box_scale = x{box_scale:.2f}")

    print("\n[Проход 4/4] Финальная 3D-реконструкция с по-кадровым 6D Bundle Adjustment:")
    frame_3d_map = {}
    total_hits = 0
    for r_idx, arcs in enumerate(all_rally_arcs):
        res_map, summaries = solve_connected_rally_arcs(arcs, K, R, T, box_scale=box_scale, refine_6d=True)
        frame_3d_map.update(res_map)
        for info in res_map.values():
            if info["hit"]:
                total_hits += 1
        arc_strs = [
            f"[{s['t_start']:.1f}-{s['t_end']:.1f}s | Z:{s['z0']:.1f}->{s['z1']:.1f}m | "
            f"v0={s['v0_kmh']:.1f} км/ч, ср={s['v_mean_kmh']:.1f} км/ч]"
            for s in summaries
        ]
        print(f"  Эпизод #{r_idx + 1:02d}: " + " -> ".join(arc_strs))

    for f_idx, (u_s, v_s, b_s) in stationary_frames.items():
        if f_idx not in frame_3d_map:
            z_bnds = ray_court_depth_bounds((u_s, v_s), K, R, T)
            pos_s = triangulate_from_depth((u_s, v_s), 0.5 * (z_bnds[0] + z_bnds[1]), K, R, T)
            frame_3d_map[f_idx] = dict(
                state=np.hstack([pos_s, [0.0, 0.0, 0.0]]),
                speed=0.0,
                v0_kmh=0.0,
                hit=False,
            )

    print("\nЗапись итогового видео и лога траектории...")
    log_rows = []
    dt = 1.0 / fps
    show_preview = SHOW_LIVE_PREVIEW

    for i in range(len(all_cands)):
        ret, frame = cap.read()
        if not ret:
            break
        t = i * dt
        chosen = chosen_list[i]
        out_of_fov = out_of_fov_list[i]
        info3d = frame_3d_map.get(i, None)

        cx = cy = box_size = conf = np.nan
        updated = False
        if chosen is not None:
            cx, cy, box_size, conf, _ = chosen
            updated = True

        if info3d is not None:
            state = info3d["state"]
            speed = info3d["speed"]
            v0_kmh = info3d.get("v0_kmh", 0.0)
            hit = info3d["hit"]
        else:
            state = None
            speed = np.nan
            v0_kmh = 0.0
            hit = False

        log_rows.append([
            t, cx, cy, box_size, conf,
            updated, hit, speed, 0.0
        ])

        draw_frame(
            frame, chosen, state,
            speed if np.isfinite(speed) else 0.0,
            v0_kmh, K, R, T, hit=hit, out_of_fov=out_of_fov
        )
        writer.write(frame)

        if show_preview:
            cv2.imshow("live", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                show_preview = False
                cv2.destroyWindow("live")

    cv2.destroyAllWindows()
    cap.release()
    writer.release()

    log_arr = np.array(log_rows, dtype=float)
    np.save(OUTPUT_TRAJECTORY, log_arr)
    print(f"\nГотово! Обработано кадров: {len(log_rows)}, обнаружено ударов: {total_hits}")


if __name__ == "__main__":
    main()