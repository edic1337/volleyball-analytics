"""
Читаем сырую траекторию (t, u, v, box_size) от ball_tracker.py, режем на
независимые сегменты полёта (segmentation.py), на каждом сегменте отдельно
вызываем batch МНК (init_state_least_squares из ball_speed.py) — получаем
ОДНО значение скорости на сегмент. Никакого continuous update после
инициализации: ошибка в одном сегменте не протекает в следующий, и
гравитация не интегрируется в моменты, когда мяч не в свободном полёте
(эти моменты просто не попадают ни в один сегмент благодаря is_static_stretch
и отсеву по MAX_TIME_GAP_S).
"""

import cv2
import numpy as np

from segmentation import segment_trajectory, estimate_segment_speed

VIDEO_PATH = "trimmed_clip_short.mp4"
TRAJ_PATH = "ball_trajectory_1.npy"
CALIB_PATH = "camera_calib_opti.npz"
OUTPUT_VIDEO = "final_speed_segments.mp4"

calib = np.load(CALIB_PATH)
K, R, T = calib["K"], calib["R"], calib["T"]
trajectory = np.load(TRAJ_PATH)  # (N, 4): t, u, v, box_size
trajectory = trajectory[np.argsort(trajectory[:, 0])]

segments = segment_trajectory(trajectory, K, R, T)
print(f"Найдено сегментов полёта: {len(segments)}")

results = []
for i, seg in enumerate(segments):
    try:
        res = estimate_segment_speed(seg, K, R, T)
    except Exception as e:
        print(f"Сегмент {i}: не удалось зафиттить ({e})")
        continue
    results.append(res)
    print(f"Сегмент {i}: t=[{res['t_start']:.2f}, {res['t_end']:.2f}]  "
          f"кадров={res['n_frames']}  скорость={res['speed_kmh']:.1f} км/ч  "
          f"RMSE={res['rmse_px']:.1f}px")

# --- Наложение результата на видео (по интервалам сегментов, без continuous state) ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
dt = 1.0 / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# индекс: для каждого кадра видео - какой сегмент (если есть) и текущая скорость
traj_dict = {round(row[0], 3): (row[1], row[2]) for row in trajectory}


def find_active_segment(t: float):
    for res in results:
        if res["t_start"] - 1e-6 <= t <= res["t_end"] + 1e-6:
            return res
    return None


current_time = 0.0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    uv = traj_dict.get(round(current_time, 3))
    active = find_active_segment(current_time)

    if uv is not None:
        u, v = uv
        cv2.circle(frame, (int(u), int(v)), 5, (0, 0, 255), -1)
        if active is not None:
            cv2.putText(frame, f"{active['speed_kmh']:.1f} km/h", (int(u) + 10, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    writer.write(frame)
    current_time += dt

cap.release()
writer.release()
print(f"\nГотово. Результат сохранён в {OUTPUT_VIDEO}")