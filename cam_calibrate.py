import cv2
import numpy as np
from scipy.optimize import minimize_scalar

VIDEO_PATH = "YOLO_result.mp4"       
FRAME_NUMBER = 767                 
OUTPUT_FILE = "camera_calib_opti.npz" 
fx, fy = 3000.0, 3000.0
cx, cy = 960.0, 540.0
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)
NET_HEIGHT = 2.43
COURT_POINTS_3D = [
    ("Угол площадки (0,0)",             (0.0,  0.0, 0.0)),
    ("Угол площадки (18,0)",            (18.0, 0.0, 0.0)),
    ("Угол площадки (18,9)",            (18.0, 9.0, 0.0)),
    ("Угол площадки (0,9)",             (0.0,  9.0, 0.0)),
    ("Пересечение линии атаки (6,0)",   (6.0,  0.0, 0.0)),
    ("Пересечение линии атаки (6,9)",   (6.0,  9.0, 0.0)),
    ("Пересечение линии атаки (12,0)",  (12.0, 0.0, 0.0)),
    ("Пересечение линии атаки (12,9)",  (12.0, 9.0, 0.0)),
    ("Верх сетки у боковой линии (9,0)", (9.0, 0.0, NET_HEIGHT)),
    ("Верх сетки у боковой линии (9,9)", (9.0, 9.0, NET_HEIGHT)),
]
clicked_points = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", current_frame)
        print(f"Точка записана: ({x}, {y})")

def get_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_number} из {video_path}")
    return frame

def reprojection_error_for_fx(fx, object_points, image_points, cx, cy):
    """Считает среднюю ошибку репроекции для заданного fx"""
    K_try = np.array([
        [fx, 0,  cx],
        [0,  fx, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K_try, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 1e6
    reproj, _ = cv2.projectPoints(object_points, rvec, tvec, K_try, DIST_COEFFS)
    reproj = reproj.reshape(-1, 2)
    return np.linalg.norm(reproj - image_points, axis=1).mean()

def find_best_fx(object_points, image_points, cx, cy, search_range=(300, 8000)):
    """Подбирает fx, минимизирующий среднюю ошибку репроекции"""
    result = minimize_scalar(
        reprojection_error_for_fx,
        bounds=search_range,
        method="bounded",
        args=(object_points, image_points, cx, cy),
        options={"xatol": 1.0}
    )
    return result.x, result.fun

def main():
    global current_frame

    current_frame = get_frame(VIDEO_PATH, FRAME_NUMBER)
    display = current_frame.copy()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    print("Кликайте точки СТРОГО в этом порядке")
    for i, (label, _) in enumerate(COURT_POINTS_3D):
        print(f"  {i+1}. {label}")
    print("\nПосле последней точки нажмите любую клавишу в окне видео\n")
    cv2.imshow("Calibration", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    n_needed = len(COURT_POINTS_3D)
    if len(clicked_points) != n_needed:
        print(f"кликнуто {len(clicked_points)} точек, а нужно {n_needed}")
        return
    image_points = np.array(clicked_points, dtype=np.float64)
    object_points = np.array([p for _, p in COURT_POINTS_3D], dtype=np.float64)
    best_fx, best_err = find_best_fx(object_points, image_points, cx, cy)
    print(f"\nПодобранное fx = {best_fx:.1f} (ошибка репроекции: {best_err:.2f} px)")
    K = np.array([
        [best_fx, 0,       cx],
        [0,       best_fx, cy],
        [0,       0,       1]
    ], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print("Проверьте порядок и точность кликов")
        return
    R, _ = cv2.Rodrigues(rvec)
    T = tvec 
    reproj, _ = cv2.projectPoints(object_points, rvec, tvec, K, DIST_COEFFS)
    reproj = reproj.reshape(-1, 2)
    errors = np.linalg.norm(reproj - image_points, axis=1)
    print(f"\nСредняя ошибка репроекции: {errors.mean():.2f} px (макс: {errors.max():.2f} px)")
    print("Хороший результат - среднее до 3-5 px")
    print("\nОшибка по каждой точке")
    for (label, _), err in zip(COURT_POINTS_3D, errors):
        marker = " " if err > errors.mean() * 2 else ""
        print(f"  {label}: {err:.2f} px{marker}")

    np.savez(OUTPUT_FILE, K=K, R=R, T=T, dist=DIST_COEFFS)
    print(f"\nСохранено: {OUTPUT_FILE}")
    print(f"R =\n{R}")
    print(f"T =\n{T.ravel()}")


if __name__ == "__main__":
    main()