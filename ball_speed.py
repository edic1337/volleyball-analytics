import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import odeint

# --- Константы ---
RHO = 1.225       # Плотность воздуха (кг/м3)
CD = 0.45         # Коэффициент сопротивления волейбольного мяча
RADIUS = 0.1045   # Радиус (м) — для Mikasa V200W абсолютно точно
MASS = 0.265      # Масса (кг) — стандарт FIVB
AREA = np.pi * RADIUS**2
K_DRAG = (RHO * CD * AREA) / (2 * MASS) 
G = 9.80665       # Стандартное ускорение свободного падения

def ball_derivatives(state, t):
    """
    Система дифференциальных уравнений полета мяча с учетом сопротивления воздуха.
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    
    # Модуль текущей 3D скорости
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Производные координат по времени (скорости)
    dx_dt = vx
    dy_dt = vy
    dz_dt = vz
    
    # Производные скоростей по времени (ускорения с торможением о воздух)
    dvx_dt = -K_DRAG * v_mag * vx
    dvy_dt = -K_DRAG * v_mag * vy
    dvz_dt = -G - K_DRAG * v_mag * vz
    
    return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]

def project_trajectory_aerodynamic(params, t_points, K_matrix, R, T):
    """
    Симулирует физику полета через ОДУ и проецирует 3D точки в 2D пиксели камеры.
    """
    X0, Y0, Z0, Vx0, Vy0, Vz0 = params
    initial_state = [float(X0), float(Y0), float(Z0), float(Vx0), float(Vy0), float(Vz0)]
    
    # Численное интегрирование системы дифференциальных уравнений
    solution = odeint(ball_derivatives, initial_state, t_points)
    
    # Извлекаем полученные 3D координаты площадки
    X = solution[:, 0]
    Y = solution[:, 1]
    Z = solution[:, 2]
    
    # Проекция на матрицу камеры
    points_3d = np.vstack((X, Y, Z))
    points_cam = R @ points_3d + T
    points_proj = K_matrix @ points_cam
    
    # Перспективное деление (u, v в пикселях)
    u_calc = points_proj[0, :] / points_proj[2, :]
    v_calc = points_proj[1, :] / points_proj[2, :]
    
    return u_calc, v_calc

def residuals(params, t, u_obs, v_obs, K, R, T):
    """
    Функция расчета остатков (ошибки репроекции) для оптимизатора least_squares.
    """
    # ИСПРАВЛЕНО: вызов функции с правильным именем
    u_calc, v_calc = project_trajectory_aerodynamic(params, t, K, R, T)
    
    error_u = u_obs - u_calc
    error_v = v_obs - v_calc
    
    # Возвращаем плоский массив ошибок, как требует least_squares
    return np.hstack((error_u, error_v))


# Фокусное расстояние и оптический центр (примерные значения для 4K видео)
fx, fy = 3000.0, 3000.0
cx, cy = 960.0, 540.0
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
])
# Поворот камеры (R) и Смещение (T) - получают через калибровку (cv2.solvePnP)
# камера смотрит под углом вниз и повернута
theta_x = np.radians(110) # Наклон вниз
R = np.array([
    [1, 0, 0],
    [0, np.cos(theta_x), -np.sin(theta_x)],
    [0, np.sin(theta_x),  np.cos(theta_x)]
])
T = np.array([[0], [-2], [5]]) # Камера висит на высоте 5 метров, смещена назад на 2 метра


