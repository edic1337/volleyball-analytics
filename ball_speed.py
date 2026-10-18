"""
ball_speed.py — Многопроходная 3D-реконструкция траектории и скорости волейбольного мяча.
Включает:
  - Робастную оценку тренда глубины по боксам (z_box_start -> z_box_end) для точных подач,
  - Защиту от ложного продольного разгона (99 км/ч) на коротких/небаллистических участках,
  - Глобальную оптимизацию связанных дуг с по-кадровой 2D-репроекцией и 6D Bundle Adjustment.
"""
import numpy as np
from scipy.optimize import least_squares

# Физические константы
RHO = 1.225
CD = 0.45
RADIUS = 0.1045      # радиус волейбольного мяча (м)
MASS = 0.270         # масса мяча (кг)
AREA = np.pi * RADIUS ** 2
K_DRAG = (RHO * CD * AREA) / (2.0 * MASS)
G = 9.80665

# Габариты игрового пространства вокруг площадки 18x9 м
COURT_MIN = np.array([-4.5, -3.5, 0.0], dtype=float)
COURT_MAX = np.array([22.5, 12.5, 10.5], dtype=float)


def ball_accel(vel):
    vx, vy, vz = vel
    v = np.hypot(np.hypot(vx, vy), vz)
    return np.array([
        -K_DRAG * v * vx,
        -K_DRAG * v * vy,
        -G - K_DRAG * v * vz
    ], dtype=float)


def f_dynamics(state):
    vel = state[3:]
    return np.hstack([vel, ball_accel(vel)])


def rk4_step(state, dt):
    k1 = f_dynamics(state)
    k2 = f_dynamics(state + 0.5 * dt * k1)
    k3 = f_dynamics(state + 0.5 * dt * k2)
    k4 = f_dynamics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_trajectory(state0, times):
    states = np.zeros((len(times), 6), dtype=float)
    curr = np.asarray(state0, dtype=float).copy()
    states[0] = curr
    curr_t = float(times[0])
    for i in range(1, len(times)):
        dt = float(times[i] - curr_t)
        if dt > 1e-6:
            n_sub = max(1, int(np.ceil(dt / 0.025)))
            dt_sub = dt / n_sub
            for _ in range(n_sub):
                curr = rk4_step(curr, dt_sub)
        states[i] = curr
        curr_t = float(times[i])
    return states


def project(state_or_pos, K, R, T):
    p = np.asarray(state_or_pos[:3], dtype=float).reshape(3, 1)
    pc = R @ p + T
    z_cam = float(pc[2, 0])
    z_safe = z_cam if abs(z_cam) > 1e-4 else 1e-4
    proj = K @ pc
    return np.array([float(proj[0, 0] / z_safe), float(proj[1, 0] / z_safe)]), z_cam


def project_points(pts_3d, K, R, T):
    """Быстрая векторизованная 2D-проекция массива 3D-точек размера (N, 3)."""
    pts = np.asarray(pts_3d[:, :3], dtype=float)
    pc = (R @ pts.T) + T.reshape(3, 1)
    z_cam = pc[2, :]
    z_safe = np.where(np.abs(z_cam) > 1e-4, z_cam, 1e-4)
    proj = K @ pc
    uv = np.vstack([proj[0, :] / z_safe, proj[1, :] / z_safe]).T
    return uv, z_cam


def triangulate_from_depth(uv, z_cam, K, R, T):
    u, v = uv
    pix = np.array([u, v, 1.0], dtype=float)
    pc = float(z_cam) * (np.linalg.inv(K) @ pix)
    return R.T @ (pc - T.flatten())


def court_depth_stats(K, R, T):
    """Возвращает (z_min_court, z_center_court, z_max_court) по геометрии площадки 18x9 м."""
    pts = np.array([
        [0.0, 0.0, 1.0], [18.0, 0.0, 1.0], [18.0, 9.0, 1.0], [0.0, 9.0, 1.0],
        [9.0, 4.5, 2.0], [3.0, 4.5, 2.0], [15.0, 4.5, 2.0]
    ], dtype=float)
    pc = (R @ pts.T) + T.reshape(3, 1)
    zs = pc[2, :]
    z_center = float(zs[4])
    z_min = float(max(4.2, np.min(zs) * 0.78))
    z_max = float(min(38.0, np.max(zs) * 1.22))
    return z_min, z_center, z_max


def ray_court_depth_bounds(uv, K, R, T):
    """Интервал глубин [z_min, z_max] вдоль луча (u, v), попадающий в игровой объем."""
    z_cmin, _, z_cmax = court_depth_stats(K, R, T)
    u, v = uv
    C = -(R.T @ T.flatten())
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    d = R.T @ ray_cam

    z_low, z_high = z_cmin, z_cmax
    for axis in range(3):
        if abs(d[axis]) >= 1e-6:
            z1 = (COURT_MIN[axis] - C[axis]) / d[axis]
            z2 = (COURT_MAX[axis] - C[axis]) / d[axis]
            if z1 > z2:
                z1, z2 = z2, z1
            z_low = max(z_low, z1)
            z_high = min(z_high, z2)

    if z_low >= z_high - 1.0:
        return z_cmin, z_cmax
    return float(z_low), float(z_high)


def ray_depth_at_world_height(uv, target_z_world, K, R, T):
    """Находит глубину z_cam вдоль луча (u, v), при которой мировая высота равна target_z_world."""
    u, v = uv
    C = -(R.T @ T.flatten())
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    d = R.T @ ray_cam
    if abs(d[2]) < 0.035:
        return None
    z_cam = (target_z_world - C[2]) / d[2]
    bnds = ray_court_depth_bounds(uv, K, R, T)
    if bnds[0] * 0.9 <= z_cam <= bnds[1] * 1.1:
        return float(np.clip(z_cam, bnds[0], bnds[1]))
    return None


def solve_ballistic_v0(pos0, pos1, dt, n_iters=6):
    """
    Точный баллистический солвер (метод пристрелки с учетом квадратичного сопротивления воздуха).
    """
    dt = max(float(dt), 1e-3)
    p0 = np.asarray(pos0, dtype=float)
    p1 = np.asarray(pos1, dtype=float)
    dp = p1 - p0
    g_vec = np.array([0.0, 0.0, -G], dtype=float)

    dist = float(np.linalg.norm(dp))
    drag_factor = 1.0 + 0.52 * K_DRAG * dist
    v0 = ((dp - 0.5 * g_vec * (dt ** 2)) / dt) * drag_factor

    times = np.array([0.0, dt])
    for _ in range(n_iters):
        st_end = integrate_trajectory(np.hstack([p0, v0]), times)[-1]
        err_p = p1 - st_end[:3]
        if float(np.linalg.norm(err_p)) < 1e-4:
            break
        v0 += (err_p / dt) * (1.0 + 0.35 * K_DRAG * dist)
    return v0


def _robust_endpoints_uv(times, u, v):
    n = len(times)
    if n <= 2:
        return float(u[0]), float(v[0]), float(u[-1]), float(v[-1])

    k_edge = min(n, 5)
    deg_edge = 1 if k_edge <= 3 else 2

    t_s = times[:k_edge] - times[0]
    u_start = float(np.polyval(np.polyfit(t_s, u[:k_edge], deg_edge), 0.0))
    v_start = float(np.polyval(np.polyfit(t_s, v[:k_edge], deg_edge), 0.0))

    t_e = times[-k_edge:] - times[-1]
    u_end = float(np.polyval(np.polyfit(t_e, u[-k_edge:], deg_edge), 0.0))
    v_end = float(np.polyval(np.polyfit(t_e, v[-k_edge:], deg_edge), 0.0))

    u_start = float(np.clip(u_start, u[0] - 4.0, u[0] + 4.0))
    v_start = float(np.clip(v_start, v[0] - 4.0, v[0] + 4.0))
    u_end = float(np.clip(u_end, u[-1] - 4.0, u[-1] + 4.0))
    v_end = float(np.clip(v_end, v[-1] - 4.0, v[-1] + 4.0))
    return u_start, v_start, u_end, v_end


def analyze_2d_arc_kinematics(arc_rows, K, R, T):
    """
    Извлекает из 2D-дуги:
      1. Локально точные координаты концов (u_start, v_start, u_end, v_end) и 2D-размах span_2d.
      2. Вертикальное ускорение a_v и гравитационную глубину z_grav.
      3. Робастный линейный тренд глубины по размеру бокса (z_box_start -> z_box_end).
    """
    arr = np.asarray(arc_rows, dtype=float)
    times = arr[:, 0]
    u = arr[:, 1]
    v = arr[:, 2]
    boxes = arr[:, 3]
    confs = arr[:, 4] if arr.shape[1] > 4 else np.ones(len(arr))

    t_mid = 0.5 * (times[0] + times[-1])
    tau = times - t_mid
    dt_total = max(float(times[-1] - times[0]), 1e-2)
    span_2d = float(np.hypot(u.max() - u.min(), v.max() - v.min()))

    deg = 2 if len(arr) >= 5 else 1
    weights = np.clip(confs, 0.25, 1.0)
    poly_u = np.polyfit(tau, u, deg, w=weights)
    poly_v = np.polyfit(tau, v, deg, w=weights)

    u_start, v_start, u_end, v_end = _robust_endpoints_uv(times, u, v)
    u_mid = float(np.polyval(poly_u, 0.0))
    v_mid = float(np.polyval(poly_v, 0.0))
    a_v = 2.0 * float(poly_v[0]) if deg == 2 else 0.0

    # 1. Гравитационная глубина по кривизне параболы
    z_grav = None
    grav_conf = 0.0
    if deg == 2 and dt_total >= 0.34 and len(arr) >= 7:
        sag_px = 0.125 * a_v * (dt_total ** 2)
        v_fit_err = float(np.median(np.abs(np.polyval(poly_v, tau) - v)))

        if a_v > 165.0 and sag_px > 18.0 and v_fit_err < 14.0:
            g_cam = R @ np.array([0.0, 0.0, -G], dtype=float)
            fy = float(K[1, 1])
            cy = float(K[1, 2])
            num_grav = abs(fy * g_cam[1] - (v_mid - cy) * g_cam[2])
            z_g = num_grav / a_v
            bnds = ray_court_depth_bounds((u_mid, v_mid), K, R, T)
            if bnds[0] * 0.78 <= z_g <= bnds[1] * 1.22:
                z_grav = float(np.clip(z_g, bnds[0], bnds[1]))
                grav_conf = float(np.clip(sag_px / 55.0, 0.25, 1.6))

    # 2. По-кадровая глубина из бокса и тренд (z_box_start -> z_box_end)
    f_eff = 0.5 * (K[0, 0] + K[1, 1])
    valid_t = []
    raw_box_z = []
    for i in range(len(arr)):
        if boxes[i] > 5.0:
            eff_b = max(float(boxes[i]), 5.0)
            z_i = f_eff * (2.0 * RADIUS) / eff_b
            valid_t.append(float(times[i] - times[0]))
            raw_box_z.append(z_i)

    z_box_raw = float(np.median(raw_box_z)) if raw_box_z else None
    z_box_start = z_box_raw
    z_box_end = z_box_raw
    box_trend_conf = 0.0

    # Тренд глубины по боксу считаем ТОЛЬКО если мяч реально летит по экрану (span_2d > 85 px)
    # и имеет положительное гравитационное ускорение (защита от рук игрока, перекрывающих мяч!)
    if len(raw_box_z) >= 5 and dt_total >= 0.22 and span_2d > 85.0 and a_v > 90.0:
        vt = np.asarray(valid_t, dtype=float)
        vz = np.asarray(raw_box_z, dtype=float)
        slopes = []
        for i in range(len(vt)):
            for j in range(i + 1, len(vt)):
                if vt[j] - vt[i] >= 0.08:
                    slopes.append((vz[j] - vz[i]) / (vt[j] - vt[i]))
        if slopes:
            slope = float(np.clip(np.median(slopes), -25.0, 25.0))
            intercept = float(np.median(vz - slope * vt))
            z_box_start = float(np.clip(intercept, 4.0, 38.0))
            z_box_end = float(np.clip(intercept + slope * dt_total, 4.0, 38.0))
            dz_span = abs(z_box_end - z_box_start)
            fit_mad = float(np.median(np.abs((intercept + slope * vt) - vz)))
            if dz_span > 2.0 and fit_mad < max(2.4, 0.42 * dz_span):
                box_trend_conf = float(np.clip(dz_span / 6.0, 0.3, 1.4))

    return dict(
        u_start=u_start, v_start=v_start,
        u_end=u_end, v_end=v_end,
        u_mid=u_mid, v_mid=v_mid,
        dt_total=dt_total,
        span_2d=span_2d,
        a_v=a_v,
        z_grav=z_grav, grav_conf=grav_conf,
        z_box_raw=z_box_raw,
        z_box_start=z_box_start,
        z_box_end=z_box_end,
        box_trend_conf=box_trend_conf,
        raw_box_z=np.asarray(raw_box_z, dtype=float),
    )


def calibrate_box_scale(all_rally_arcs, K, R, T):
    _, z_center_court, _ = court_depth_stats(K, R, T)
    grav_ratios = []
    all_raw_z = []

    for arcs in all_rally_arcs:
        for arc in arcs:
            info = analyze_2d_arc_kinematics(arc, K, R, T)
            if info["z_box_raw"] is not None:
                all_raw_z.append(info["z_box_raw"])
                if info["z_grav"] is not None and info["grav_conf"] >= 0.55:
                    grav_ratios.append(info["z_grav"] / info["z_box_raw"])

    if len(grav_ratios) >= 2:
        scale = float(np.median(grav_ratios))
    elif all_raw_z:
        scale = z_center_court / float(np.median(all_raw_z))
    else:
        scale = 1.0

    return float(np.clip(scale, 0.95, 1.65))


def _refine_single_arc_6d(arc, p0_init, p1_init, K, R, T):
    """
    Этап 2B: Локальная 6D-доводка дуги [X0, Y0, Z0, Vx0, Vy0, Vz0] по 2D-детекциям YOLO.
    Жестко удерживает 3D-концы дуги вблизи (p0_init, p1_init), чтобы желтая точка шла по центру
    бокса YOLO, но не могла накрутить ложную продольную скорость вдоль луча камеры.
    """
    arr = np.asarray(arc, dtype=float)
    times = arr[:, 0]
    u_obs = arr[:, 1]
    v_obs = arr[:, 2]
    confs = np.clip(arr[:, 4], 0.25, 1.0) if arr.shape[1] > 4 else np.ones(len(arr))
    dt_s = max(float(times[-1] - times[0]), 1e-3)

    v0_init = solve_ballistic_v0(p0_init, p1_init, dt_s)
    s0_init = np.hstack([p0_init, v0_init])

    # Узкий коридор доводки: исключает искусственный разгон вдоль луча камеры
    lb = np.hstack([p0_init - np.array([0.45, 0.45, 0.35]), v0_init - np.array([3.2, 3.2, 2.8])])
    ub = np.hstack([p0_init + np.array([0.45, 0.45, 0.35]), v0_init + np.array([3.2, 3.2, 2.8])])

    def res_6d(s0):
        states = integrate_trajectory(s0, times)
        uv_proj, _ = project_points(states, K, R, T)
        err_u = (uv_proj[:, 0] - u_obs) * confs / 2.2
        err_v = (uv_proj[:, 1] - v_obs) * confs / 2.2
        err_p0 = (states[0, :3] - p0_init) / 0.18
        err_p1 = (states[-1, :3] - p1_init) / 0.22
        err_v0 = (s0[3:] - v0_init) / 1.8
        return np.hstack([err_u, err_v, err_p0, err_p1, err_v0])

    opt = least_squares(res_6d, s0_init, bounds=(lb, ub), loss="soft_l1", max_nfev=100)
    return opt.x


def solve_connected_rally_arcs(arcs, K, R, T, box_scale=1.0, refine_6d=True):
    n_arcs = len(arcs)
    if n_arcs == 0:
        return {}, []

    _, z_center_court, _ = court_depth_stats(K, R, T)
    arc_infos = [analyze_2d_arc_kinematics(arc, K, R, T) for arc in arcs]

    base_uvs = [(arc_infos[0]["u_start"], arc_infos[0]["v_start"])]
    for s in range(n_arcs):
        if s < n_arcs - 1:
            u_m = 0.5 * (arc_infos[s]["u_end"] + arc_infos[s + 1]["u_start"])
            v_m = 0.5 * (arc_infos[s]["v_end"] + arc_infos[s + 1]["v_start"])
            base_uvs.append((u_m, v_m))
        else:
            base_uvs.append((arc_infos[s]["u_end"], arc_infos[s]["v_end"]))

    n_nodes = len(base_uvs)
    node_bounds = [ray_court_depth_bounds(uv, K, R, T) for uv in base_uvs]

    node_target_z = np.full(n_nodes, z_center_court, dtype=float)
    node_weights = np.zeros(n_nodes, dtype=float)

    for s, info in enumerate(arc_infos):
        z_s = z_center_court
        z_e = z_center_court
        if info["z_box_start"] is not None:
            z_s = info["z_box_start"] * box_scale
            z_e = info["z_box_end"] * box_scale
        if info["z_grav"] is not None:
            wg = info["grav_conf"] / (info["grav_conf"] + 0.75)
            dz_box = (z_e - z_s) if info["box_trend_conf"] > 0.35 else 0.0
            z_mid_target = wg * info["z_grav"] + (1.0 - wg) * 0.5 * (z_s + z_e)
            z_s = z_mid_target - 0.5 * dz_box
            z_e = z_mid_target + 0.5 * dz_box

        b0, b1 = node_bounds[s], node_bounds[s + 1]
        z_s = float(np.clip(z_s, b0[0], b0[1]))
        z_e = float(np.clip(z_e, b1[0], b1[1]))

        w_arc = 1.0 + info["grav_conf"] + 0.6 * info["box_trend_conf"]
        if node_weights[s] == 0.0:
            node_target_z[s] = z_s
            node_weights[s] = w_arc
        else:
            node_target_z[s] = (node_target_z[s] * node_weights[s] + z_s * w_arc) / (node_weights[s] + w_arc)
            node_weights[s] += w_arc

        node_target_z[s + 1] = z_e
        node_weights[s + 1] = w_arc

    # Уточнение дальней точки приема для длинных подач/атак (при span_2d > 140 px и падении вниз)
    for s, info in enumerate(arc_infos):
        if info["dt_total"] >= 0.32 and info["span_2d"] > 140.0 and info["v_end"] > info["v_start"] + 35.0:
            z_rec = ray_depth_at_world_height(base_uvs[s + 1], 1.25, K, R, T)
            if z_rec is not None and z_rec > node_target_z[s] + 3.5:
                node_target_z[s + 1] = 0.55 * node_target_z[s + 1] + 0.45 * z_rec

    x_init = np.zeros(3 * n_nodes, dtype=float)
    lb = np.zeros(3 * n_nodes, dtype=float)
    ub = np.zeros(3 * n_nodes, dtype=float)
    for m in range(n_nodes):
        b_lo, b_hi = node_bounds[m]
        x_init[3 * m] = np.clip(node_target_z[m], b_lo + 0.05, b_hi - 0.05)
        lb[3 * m], ub[3 * m] = b_lo, b_hi
        lb[3 * m + 1], ub[3 * m + 1] = -8.0, 8.0
        lb[3 * m + 2], ub[3 * m + 2] = -8.0, 8.0

    arc_samples = []
    for arc in arcs:
        arr = np.asarray(arc, dtype=float)
        step = max(1, len(arr) // 12)
        idx = np.unique(np.append(np.arange(0, len(arr), step), len(arr) - 1))
        arc_samples.append(arr[idx])

    def residuals(x_vars):
        res = []
        z_nodes = x_vars[0::3]
        du_nodes = x_vars[1::3]
        dv_nodes = x_vars[2::3]

        pts = []
        for m in range(n_nodes):
            uv_m = (base_uvs[m][0] + du_nodes[m], base_uvs[m][1] + dv_nodes[m])
            pt = triangulate_from_depth(uv_m, z_nodes[m], K, R, T)
            pts.append(pt)

            err_x = min(0.0, pt[0] - (-2.5)) + max(0.0, pt[0] - 20.5)
            err_y = min(0.0, pt[1] - (-2.0)) + max(0.0, pt[1] - 11.0)
            err_z = min(0.0, pt[2] - 0.35) * 3.0 + max(0.0, pt[2] - 3.55) * 2.8
            res.extend([err_x * 1.8, err_y * 1.8, err_z])
            res.extend([du_nodes[m] / 4.0, dv_nodes[m] / 4.0])

        for s, info in enumerate(arc_infos):
            dt_s = info["dt_total"]
            p0, p1 = pts[s], pts[s + 1]
            z0, z1 = z_nodes[s], z_nodes[s + 1]
            z_mean = 0.5 * (z0 + z1)

            if info["z_grav"] is not None:
                sigma_g = 1.9 / max(info["grav_conf"], 0.4)
                res.append((z_mean - info["z_grav"]) / sigma_g)
            else:
                res.append(0.0)

            if info["z_box_start"] is not None:
                zb_s = info["z_box_start"] * box_scale
                zb_e = info["z_box_end"] * box_scale
                sigma_b = 2.5 if info["box_trend_conf"] > 0.3 else 3.4
                res.append((z0 - zb_s) / sigma_b)
                res.append((z1 - zb_e) / sigma_b)
                v_rad_box = (zb_e - zb_s) / max(dt_s, 0.12)
            else:
                res.extend([0.0, 0.0])
                v_rad_box = 0.0

            # Защита от ложных 99 км/ч вдоль луча при сохранении честной скорости на подачах:
            v_radial = (z1 - z0) / max(dt_s, 0.12)
            if info["span_2d"] < 110.0 or info["a_v"] < 110.0:
                # Короткое 2D-движение или слабая кривизна: жестко гасим перспективный рычаг
                res.append(v_radial / 3.2)
            elif info["box_trend_conf"] > 0.3:
                # Есть выраженный тренд бокса (подача/атака): привязываем продольную скорость к тренду бокса
                res.append((v_radial - v_rad_box) / 6.5)
            else:
                # Обычная дуга без сильного изменения размера бокса: умеренная регуляризация глубины
                res.append(v_radial / 5.8)

            v0 = solve_ballistic_v0(p0, p1, dt_s, n_iters=4)
            spd0 = float(np.linalg.norm(v0))
            res.append(max(0.0, spd0 - 33.0) * 3.5)

            samp = arc_samples[s]
            t_rel = samp[:, 0] - samp[0, 0]
            states_s = integrate_trajectory(np.hstack([p0, v0]), t_rel)
            uv_proj, _ = project_points(states_s, K, R, T)

            # Если дуга имеет нормальную гравитацию, плотно притягиваем 2D-проекцию
            reproj_scale = 3.2 if info["a_v"] > 120.0 else 6.5
            u_err = (uv_proj[:, 0] - samp[:, 1]) / reproj_scale
            v_err = (uv_proj[:, 1] - samp[:, 2]) / reproj_scale
            res.extend(u_err.tolist())
            res.extend(v_err.tolist())

        return np.asarray(res, dtype=float)

    opt = least_squares(
        residuals, x_init, bounds=(lb, ub),
        loss="soft_l1", f_scale=1.5, max_nfev=250
    )
    x_opt = opt.x
    z_opt = x_opt[0::3]
    du_opt = x_opt[1::3]
    dv_opt = x_opt[2::3]
    pts_opt = [
        triangulate_from_depth((base_uvs[m][0] + du_opt[m], base_uvs[m][1] + dv_opt[m]), z_opt[m], K, R, T)
        for m in range(n_nodes)
    ]

    frame_results = {}
    arc_summaries = []

    for s, arc in enumerate(arcs):
        p0 = pts_opt[s]
        p1 = pts_opt[s + 1]
        f_start = int(arc[0][5])
        f_end = int(arc[-1][5])
        t_start = float(arc[0][0])
        t_end = float(arc[-1][0])
        dt_s = max(t_end - t_start, 1e-3)

        if refine_6d and len(arc) >= 5 and arc_infos[s]["a_v"] > 110.0:
            s0_opt = _refine_single_arc_6d(arc, p0, p1, K, R, T)
        else:
            v0 = solve_ballistic_v0(p0, p1, dt_s, n_iters=6)
            s0_opt = np.hstack([p0, v0])

        n_frames_arc = max(1, f_end - f_start + 1)
        all_times = np.linspace(t_start, t_end, n_frames_arc)
        all_states = integrate_trajectory(s0_opt, all_times)

        speeds = np.linalg.norm(all_states[:, 3:], axis=1)
        v0_kmh = float(speeds[0] * 3.6)
        v_mean_kmh = float(np.mean(speeds) * 3.6)

        _, z_cam_arc = project_points(all_states, K, R, T)
        arc_summaries.append(dict(
            t_start=t_start, t_end=t_end,
            z0=float(z_cam_arc[0]), z1=float(z_cam_arc[-1]),
            v0_kmh=v0_kmh, v_mean_kmh=v_mean_kmh,
        ))

        for k in range(n_frames_arc):
            f_idx = f_start + k
            st = all_states[k]
            spd = float(speeds[k])
            is_hit = (s > 0 and k == 0)
            frame_results[f_idx] = dict(
                state=st,
                speed=spd,
                v0_kmh=v0_kmh,
                hit=is_hit,
            )

    return frame_results, arc_summaries