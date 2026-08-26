import cv2
import numpy as np
from ball_speed import BallEKF, triangulate_from_depth, project

VIDEO_PATH = "trimmed_clip_short_2.mp4"
TRAJ_PATH = "ball_trajectory_2.npy"
CALIB_PATH = "camera_calib_opti.npz"
OUTPUT_VIDEO = "final_speed_2.mp4"

calib = np.load(CALIB_PATH)
K, R, T = calib['K'], calib['R'], calib['T']
trajectory = np.load(TRAJ_PATH)
traj_dict = {round(row[0], 3): (row[1], row[2]) for row in trajectory}

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
dt = 1.0 / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

ekf = None
is_ekf_initialized = False
frames_lost = 0
MAX_LOST = 10
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def get_start_3d_position(frame, ball_uv, K, R, T, forward_offset=0.25):
    """
    Ставит кадр на паузу, ждет клика по ногам подающего и 
    возвращает точную 3D-координату мяча (учитывая подброс вперед).
    """
    global clicked_point
    clicked_point = None
    
    window_name = "Click on Player's Feet (Press SPACE to skip)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Рисуем маркер на мяче, чтобы было понятно, чей это подброс
    display_frame = frame.copy()
    cv2.circle(display_frame, (int(ball_uv[0]), int(ball_uv[1])), 8, (0, 255, 0), 2)
    cv2.putText(display_frame, "Click on server's feet", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Ждем клика или нажатия пробела
    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if clicked_point is not None:
            break
        if key == 32:
            break
            
    cv2.destroyWindow(window_name)
    
    if clicked_point is None:
        return None
        
    u_feet, v_feet = clicked_point
    
    # Координаты центра камеры в мире (C = -R^T * T)
    C = -R.T @ T.reshape(3, 1)
    
    # Находим 3D-координаты ног (пересечение луча с полом Z = 0)
    pix_feet = np.array([u_feet, v_feet, 1.0]).reshape(3, 1)
    d_feet = R.T @ np.linalg.inv(K) @ pix_feet  # Вектор луча ног
    
    t_feet = (0.0 - C[2, 0]) / d_feet[2, 0]     # Параметр луча для пола
    feet_3d = C + t_feet * d_feet
    x_feet = feet_3d[0, 0]
    
    # Смещаем плоскость мяча на 25 см вперед
    direction = 1.0 if x_feet < 9.0 else -1.0
    x_ball_plane = x_feet + (forward_offset * direction)
    
    # Проецируем луч мяча на смещенную плоскость X
    pix_ball = np.array([ball_uv[0], ball_uv[1], 1.0]).reshape(3, 1)
    d_ball = R.T @ np.linalg.inv(K) @ pix_ball  # Вектор луча мяча
    
    t_ball = (x_ball_plane - C[0, 0]) / d_ball[0, 0]
    ball_3d = C + t_ball * d_ball
    
    return ball_3d.flatten()

current_time = 0.0
print("Наложение 3D-скорости на видео...")

while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_key = round(current_time, 3)
        uv = traj_dict.get(time_key)

        if uv is not None:
            # === МЯЧ ЕСТЬ НА КАДРЕ ===
            frames_lost = 0
            u, v = uv
            
            # 1. Проверяем, запущен ли фильтр
            if not is_ekf_initialized:
                # Запрашиваем 3D координаты старта через ручной клик
                pos0 = get_start_3d_position(frame, (u, v), K, R, T, forward_offset=0.25)
                
                if pos0 is not None:
                    ekf = BallEKF(K, R, T, dt)
                    ekf.init_state(current_time, pos0, vel0_guess=(0.0, 0.0, 0.0))
                    is_ekf_initialized = True
                    print(f"Старт 3D захвачен: X={pos0[0]:.2f}, Y={pos0[1]:.2f}, Z={pos0[2]:.2f}")
                else:
                    # Если нажали пробел, просто пропускаем этот кадр. 
                    # Физика не запустится, скрипт пойдет дальше.
                    pass
            
            # 2. Если фильтр работает (или только что запустился) - считаем скорость
            if is_ekf_initialized:
                state = ekf.step(current_time, uv=uv)
                speed_kmh = float(np.linalg.norm(state[3:])) * 3.6
                
                # Рисуем зеленую надпись скорости
                cv2.circle(frame, (int(u), int(v)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{speed_kmh:.1f} km/h", (int(u) + 10, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # === МЯЧА НЕТ НА КАДРЕ (СЛЕПАЯ ЗОНА) ===
            frames_lost += 1
            if is_ekf_initialized:
                if frames_lost <= MAX_LOST:
                    # Продолжаем двигать мяч по инерции (желтый маркер)
                    state = ekf.step(current_time, uv=None)
                    speed_kmh = float(np.linalg.norm(state[3:])) * 3.6
                    
                    pred_uv, _ = project(state, K, R, T)
                    pred_u, pred_v = int(pred_uv[0]), int(pred_uv[1])
                    
                    cv2.circle(frame, (pred_u, pred_v), 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"PRED: {speed_kmh:.1f} km/h", (pred_u + 10, pred_v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Ждали слишком долго, глушим физику до следующей подачи
                    is_ekf_initialized = False
                    print("Трек сброшен по таймауту.")

        writer.write(frame)
        current_time += dt

cap.release()
writer.release()
print(f"Готово! Результат сохранен в {OUTPUT_VIDEO}")