"""
main.py — совместная работа YOLO-детектора и физической EKF-модели полёта
мяча для непрерывного точного определения скорости.
  1. YOLO на каждом кадре может дать НЕСКОЛЬКО кандидатов (мяч, похожие
     объекты, шум). Раньше "свой" кандидат выбирался по фиксированному
     пиксельному радиусу от last_center (ball_tracker.py, MAX_PIXEL_JUMP) —
     это ломалось, если last_center "залипал" на N кадров потери детекции,
     а мяч успевал улететь дальше 90px.

     Здесь вместо этого на каждом кадре сначала делается PREDICT-шаг EKF
     (физика: RK4 по гравитации + сопротивлению воздуха) БЕЗ коммита в
     состояние фильтра — только чтобы получить предсказанную позицию (u,v)
     и её ковариацию S = H P H^T + R. Кандидаты YOLO гейтятся по
     Махаланобис-дистанции к этому предсказанию (chi2-тест), а среди
     прошедших гейт выбирается лучший с учётом confidence YOLO. Гейт
     "дышит" вместе с фильтром: после нескольких кадров без детекций P
     растёт (через Q в predict), и радиус поиска сам расширяется —
     не нужно вручную растягивать константу.

  2. Если ни один кандидат не прошёл гейт (окклюзия, мяч за кадром,
     смазан) — детекции в этом кадре просто нет. Состояние фильтра всё
     равно обновляется физикой ("движок вместо трекера" — идея, которая
     была описана в доктстринге ball_speed.py, но не была подключена к
     трекеру). Скорость не теряется и не "зависает" на время потери.

  3. Удары (руки/пол/сетка) ловятся не пост-хок сегментацией по времени,
     а онлайн — через NIS-гейтинг внутри BallEKF.update на каждом
     реальном обновлении.

  4. conf YOLO используется дважды: как вес при выборе кандидата среди
     нескольких, прошедших гейт, и как масштаб R (шума измерения) внутри
     самого EKF-апдейта (meas_conf).

  5. Скорость на каждый кадр = ||v|| из состояния EKF — непрерывный ряд,
     без разрывов на границах "сегментов".

Требует в одной директории: ball_speed.py (BallEKF, физика, проекция,
батч-МНК init_state_robust — используется только один раз, для
инициализации фильтра по первым N точкам).
"""

import cv2
import numpy as np
from ultralytics import YOLO

from ball_speed import (
    BallEKF, project, measurement_jacobian, _h,
    rk4_step, jacobian_f, init_state_robust,
)

VIDEO_PATH = "trimmed_clip.mp4"
YOLO_WEIGHTS = "runs/detect/train-18/weights/best.pt"
CALIB_PATH = "camera_calib_opti.npz"
OUTPUT_VIDEO = "final_speed_ekf.mp4"
OUTPUT_TRAJECTORY = "ball_trajectory_ekf.npy"
CONF_THRESHOLD = 0.26
IOU_THRESHOLD = 0.45
IMG_SIZE = 1280
N_INIT_POINTS = 15               # точек для батч-МНК инициализации EKF (нужна оценка глубины по боксу)
INIT_BOOTSTRAP_GATE_PX = 120.0   # грубый гейт на этапе bootstrap, пока EKF ещё не запущен
GATE_CHI2 = 9.21                 # 99% для 2 ст.св. — гейт кандидатов YOLO по (u,v) предсказанию
MEAS_PX_STD = 3.0
MEAS_DEPTH_STD = 4.0


def load_calib(path):
    calib = np.load(path)
    return calib["K"], calib["R"], calib["T"]


def yolo_detect(model, frame):
    """Возвращает список кандидатов (cx, cy, box_size, conf, bbox)."""
    results = model.predict(
        frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE, verbose=False,
    )[0]
    cands = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            box_size = min(x2 - x1, y2 - y1)
            cands.append((cx, cy, box_size, float(c),
                          (int(x1), int(y1), int(x2), int(y2))))
    return cands


def pick_bootstrap_candidate(cands, last_center):
    """До инициализации EKF гейтить нечем — выбираем как в старом
    ball_tracker.py: по близости к последней принятой точке, либо
    по максимальному conf на самом первом успешном кадре."""
    if not cands:
        return None
    if last_center is None:
        return max(cands, key=lambda c: c[3])
    best, best_dist = None, float("inf")
    for c in cands:
        d = np.hypot(c[0] - last_center[0], c[1] - last_center[1])
        if d < best_dist and d < INIT_BOOTSTRAP_GATE_PX:
            best_dist, best = d, c
    return best


def predict_peek(ekf, dt):
    """Physика-предсказание БЕЗ коммита в состояние фильтра — нужно
    только чтобы получить гейт (позицию + ковариацию) для выбора
    детекции в текущем кадре."""
    F = jacobian_f(ekf.x, dt)
    x_pred = rk4_step(ekf.x, dt)
    Q = ekf._process_noise(dt)
    P_pred = F @ ekf.P @ F.T + Q
    return x_pred, P_pred


def gate_and_pick(cands, x_pred, P_pred, K, R, T):
    """Гейтинг кандидатов YOLO по Махаланобис-дистанции к physика-
    предсказанию (вместо фиксированного пиксельного радиуса). Среди
    прошедших гейт выбирается лучший с учётом conf."""
    if not cands:
        return None, None

    h0 = _h(x_pred, K, R, T, use_depth=False)
    H = measurement_jacobian(x_pred, K, R, T, use_depth=False)
    Rmat = np.eye(2) * MEAS_PX_STD ** 2
    S = H @ P_pred @ H.T + Rmat
    S_inv = np.linalg.inv(S)

    best, best_score, best_nis = None, float("inf"), None
    for c in cands:
        y = np.array([c[0], c[1]]) - h0
        nis = float(y @ S_inv @ y)
        if nis > GATE_CHI2:
            continue
        score = nis / max(c[3], 0.05)  # штраф за низкий conf
        if score < best_score:
            best_score, best, best_nis = score, c, nis
    return best, best_nis


def draw_frame(frame, chosen, state, speed, K, R, T, hit=False, frames_lost=0):
    (u_phys, v_phys), _ = project(state, K, R, T)
    color_phys = (255, 0, 0) if chosen is None else (0, 200, 255)
    cv2.circle(frame, (int(u_phys), int(v_phys)), 4, color_phys, -1)

    if chosen is not None:
        cx, cy, box_size, conf, (x1, y1, x2, y2) = chosen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"YOLO {conf:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"PHYSICS ONLY (lost {frames_lost})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, f"{speed * 3.6:.1f} km/h", (int(u_phys) + 12, int(v_phys) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if hit:
        cv2.putText(frame, "HIT!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def main():
    K, R, T = load_calib(CALIB_PATH)
    model = YOLO(YOLO_WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_nominal = 1.0 / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    log_rows = []  # t, u, v, box_size, conf, updated(bool), hit(bool), speed(m/s)

    # ============== Фаза 1: bootstrap — копим точки, батч-МНК init EKF ==============
    bootstrap_rows = []    # [t, u, v, box_size] — только точки с валидным box_size
    bootstrap_frames = []  # (t, frame_copy, chosen)
    last_center = None
    t = 0.0
    frame_idx = 0
    ekf = None

    while ekf is None:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"Видео закончилось, не набрав {N_INIT_POINTS} точек для "
                "инициализации EKF — мяч ни разу уверенно не задетектирован."
            )
        cands = yolo_detect(model, frame)
        chosen = pick_bootstrap_candidate(cands, last_center)
        bootstrap_frames.append((t, frame.copy(), chosen))

        if chosen is not None:
            cx, cy, box_size, conf, _ = chosen
            last_center = (cx, cy)
            if box_size > 1e-3:
                bootstrap_rows.append([t, cx, cy, box_size])

        t += dt_nominal
        frame_idx += 1

        if len(bootstrap_rows) >= N_INIT_POINTS:
            state0 = init_state_robust(bootstrap_rows, K, R, T)
            ekf = BallEKF(K, R, T, dt=dt_nominal,
                          process_accel_std=1.0,
                           meas_px_std=MEAS_PX_STD, meas_depth_std=MEAS_DEPTH_STD)
            t0 = bootstrap_rows[0][0]
            ekf.init_state(t0, state0[:3], state0[3:])
            print(f"EKF инициализирован по {len(bootstrap_rows)} точкам: "
                  f"pos0={state0[:3].round(2)}  v0={state0[3:].round(2)} м/с "
                  f"(|v0|={np.linalg.norm(state0[3:]) * 3.6:.1f} км/ч)")

    # "довести" фильтр по уже просмотренным bootstrap-кадрам обычным шагом,
    # чтобы Фаза 2 продолжала с тем же непрерывным состоянием
    for ft, frame, chosen in bootstrap_frames:
        cx = cy = box_size = conf = None
        uv = None
        if chosen is not None:
            cx, cy, box_size, conf, _ = chosen
            uv = (cx, cy)
        x = ekf.step(ft, uv, box_size, conf if conf is not None else 1.0)
        speed = float(np.linalg.norm(x[3:]))
        hit = ekf.history[-1]["hit"]
        log_rows.append([ft, cx if uv else np.nan, cy if uv else np.nan,
                          box_size if box_size else np.nan,
                          conf if conf is not None else np.nan,
                          uv is not None, hit, speed])
        draw_frame(frame, chosen, x, speed, K, R, T, hit)
        writer.write(frame)

    frames_lost = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cands = yolo_detect(model, frame)

        dt = t - ekf._t_prev
        x_pred, P_pred = predict_peek(ekf, max(dt, 1e-6))
        chosen, nis = gate_and_pick(cands, x_pred, P_pred, K, R, T)

        cx = cy = box_size = conf = None
        uv = None
        if chosen is not None:
            cx, cy, box_size, conf, _ = chosen
            uv = (cx, cy)
            frames_lost = 0
        else:
            frames_lost += 1

        x = ekf.step(t, uv, box_size, conf if conf is not None else 1.0)
        speed = float(np.linalg.norm(x[3:]))
        hit = ekf.history[-1]["hit"]

        log_rows.append([t, cx if uv else np.nan, cy if uv else np.nan,
                          box_size if box_size else np.nan,
                          conf if conf is not None else np.nan,
                          uv is not None, hit, speed])

        draw_frame(frame, chosen, x, speed, K, R, T, hit, frames_lost)
        writer.write(frame)

        t += dt_nominal
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  обработано кадров: {frame_idx}  текущая скорость: {speed * 3.6:.1f} км/ч")

    cap.release()
    writer.release()

    log_arr = np.array(log_rows, dtype=float)
    np.save(OUTPUT_TRAJECTORY, log_arr)

    print(f"\nГотово. Кадров обработано: {frame_idx}.")
    print(f"Ударов / резких смен скорости обнаружено: {len(ekf.events)}")
    for ev in ekf.events:
        print(f"  t={ev['t']:.2f}s  |v| до={ev['speed_before'] * 3.6:.1f} км/ч -> "
              f"после={ev['speed_after'] * 3.6:.1f} км/ч")
    print(f"Видео сохранено: {OUTPUT_VIDEO}")
    print(f"Лог траектории+скорости сохранён: {OUTPUT_TRAJECTORY}")
    print("  колонки: t, u, v, box_size, conf, updated(bool), hit(bool), speed(m/s)")


if __name__ == "__main__":
    main()