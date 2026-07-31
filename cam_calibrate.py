import cv2
import numpy as np

# Глобальный список для хранения пикселей, по которым кликнул пользователь
image_points = []
clone_frame = None

def click_event(event, x, y, flags, param):
    """Обработчик кликов мыши по картинке"""
    global image_points, clone_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) < 4:
            image_points.append([float(x), float(y)])
            # Рисуем кружок и номер точки
            cv2.circle(clone_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(clone_frame, str(len(image_points)), (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Calibration", clone_frame)

def main():
    global clone_frame, image_points
    
    video_path = 'match.mp4' # путь к видео
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Ошибка: Не удалось прочитать видеофайл")
        return

    clone_frame = frame.copy()
    
    print("Открыто окно калибровки.")
    print("Кликни левой кнопкой мыши по 4 углам площадки СТРОГО в таком порядке:")
    print("1. Ближний ЛЕВЫЙ угол")
    print("2. Ближний ПРАВЫЙ угол")
    print("3. Дальний ПРАВЫЙ угол")
    print("4. Дальний ЛЕВЫЙ угол")
    print("После выбора 4 точек нажми любую клавишу в окне картинки.")

    cv2.imshow("Calibration", clone_frame)
    cv2.setMouseCallback("Calibration", click_event)
    
    # Ждем, пока пользователь кликнет 4 раза и нажмет любую кнопку
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(image_points) != 4:
        print("Ошибка: Нужно выбрать ровно 4 точки!")
        return

    # 1. Задаем реальные 3D-координаты углов (в метрах)
    # Порядок строго совпадает с инструкцией для кликов!
    object_points = np.array([
        [0.0, 0.0, 0.0],    # 1. Ближний левый (начало координат)
        [9.0, 0.0, 0.0],    # 2. Ближний правый (ширина 9м)
        [9.0, 18.0, 0.0],   # 3. Дальний правый (длина 18м)
        [0.0, 18.0, 0.0]    # 4. Дальний левый
    ], dtype=np.float32)

    img_points_arr = np.array(image_points, dtype=np.float32)

    # Получаем ширину и высоту из видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Вычисляем оптический центр
    cx = width / 2.0
    cy = height / 2.0

    # Задаем приблизительное фокусное расстояние 
    # (если после калибровки высота телефона Z получается неадекватной, 
    # поменять множитель 0.8 на 0.6 или 1.2)
    focal_length = width * 0.8  
    fx = focal_length
    fy = focal_length

    # Собираем итоговую матрицу K
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # 3. Решаем PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, 
        img_points_arr, 
        K, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("SolvePnP не смог найти решение.")
        return

    # Переводим вектор вращения в матрицу 3x3
    R, _ = cv2.Rodrigues(rvec)

    # Вычисляем координаты камеры в мире: -R^T * T
    camera_pos = -np.dot(R.T, tvec)

    print("\n" + "="*40)
    print("УСПЕШНАЯ КАЛИБРОВКА!")
    print("="*40)
    print("Матрица R (Rotation):\n", R)
    print("\nВектор T (Translation):\n", tvec)
    print("\n=== Координаты телефона ===")
    print(f"По оси X (Смещение вбок): {camera_pos[0][0]:.2f} метров")
    print(f"По оси Y (Удаление за линию): {camera_pos[1][0]:.2f} метров")
    print(f"По оси Z (Высота телефона): {camera_pos[2][0]:.2f} метров")
    print("="*40)

    # Раскомментируй, если хочешь сразу сохранять матрицы в файл для ball_speed.py
    # np.savez('camera_params.npz', K=K, R=R, T=tvec)
    # print("Параметры сохранены в camera_params.npz")

if __name__ == "__main__":
    main()