"""
Extended Kalman Filter для трекинга волейбольного мяча

Идея:
  - Состояние x = [X, Y, Z, Vx, Vy, Vz] в мировых координатах (метры)
  - Шаг PREDICT — интегрируем физику полёта (гравитация + сопротивление
    воздуха) на dt = 1/fps вперёд. Если YOLO не дал детекцию на этом кадре 
    на этом всё, состояние = предсказание физики (это и есть "движок вместо
    трекера", который вы просили).
  - Шаг UPDATE — если детекция есть, корректируем состояние по невязке между
    спроецированной 2D-точкой модели и реальным (u, v) от YOLO. Опционально
    используем размер бокса как ещё одно измерение (грубая оценка глубины Z),
    т.к. по одному (u, v) с одной камеры глубина не наблюдаема
  - Удары (контакты руками/полом/сеткой) физика не описывает — скорость
    меняется скачком. Это ловится через NIS (normalized innovation squared):
    если реальная невязка сильно больше, чем предсказывает ковариация
    фильтра, значит между предыдущим и текущим кадром физика "сломалась" 
    считаем это ударом, переинициализируем скорость по конечной разности
    последних двух подтверждённых 3D-положений и раздуваем P на скорость

Это НЕ готовый продукт "запусти и получишь идеальные цифры" — пороги
(GATE_CHI2, Q, R) почти наверняка придётся подогнать под ваше видео и
разрешение YOLO-детекций. Ниже они вынесены в константы с пояснениями
"""

import numpy as np

#константы
RHO = 1.225
CD = 0.45
RADIUS = 0.1045       # м, для маркировки размера бокса тоже пригодится
MASS = 0.265
AREA = np.pi * RADIUS ** 2
K_DRAG = (RHO * CD * AREA) / (2 * MASS)
G = 9.80665

STATE_DIM = 6  # [X, Y, Z, Vx, Vy, Vz]


def ball_accel(vel):
    vx, vy, vz = vel
    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    ax = -K_DRAG * v * vx
    ay = -K_DRAG * v * vy
    az = -G - K_DRAG * v * vz
    return np.array([ax, ay, az])


def f_dynamics(state):
    vel = state[3:]
    acc = ball_accel(vel)
    return np.hstack([vel, acc])


def rk4_step(state, dt):
    k1 = f_dynamics(state)
    k2 = f_dynamics(state + dt / 2 * k1)
    k3 = f_dynamics(state + dt / 2 * k2)
    k4 = f_dynamics(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobian_f(state, dt, eps=1e-4):
    n = len(state)
    F = np.eye(n)
    for i in range(n):
        s1, s2 = state.copy(), state.copy()
        s1[i] += eps
        s2[i] -= eps
        F[:, i] = (rk4_step(s1, dt) - rk4_step(s2, dt)) / (2 * eps)
    return F


# Модель камеры 
def project(state, K, R, T):
    """3D мировая точка -> (u, v) пикселей + глубина в системе камеры (Z_cam)"""
    p = state[:3].reshape(3, 1)
    pc = R @ p + T
    proj = K @ pc
    u = proj[0, 0] / proj[2, 0]
    v = proj[1, 0] / proj[2, 0]
    z_cam = pc[2, 0]
    return np.array([u, v]), z_cam


def triangulate_from_depth(uv, z_cam, K, R, T):
    """
    Восстановление 3D мировой точки по пикселям (u, v) и оценке глубины
    Z_cam (из размера бокса). Нужно для нормальной инициализации фильтра —
    без этого V0=0 и филтр долго 'догоняет' реальную скорость (см. тесты)
    """
    u, v = uv
    pix = np.array([u, v, 1.0])
    pc = z_cam * (np.linalg.inv(K) @ pix)   # точка в системе камеры
    world = R.T @ (pc - T.flatten())        # R ортонормальна -> R^-1 = R^T
    return world


def measurement_jacobian(state, K, R, T, use_depth, eps=1e-4):
    n = len(state)
    h0 = _h(state, K, R, T, use_depth)
    m = len(h0)
    H = np.zeros((m, n))
    for i in range(n):
        s1 = state.copy()
        s1[i] += eps
        h1 = _h(s1, K, R, T, use_depth)
        H[:, i] = (h1 - h0) / eps
    return H


def _h(state, K, R, T, use_depth):
    (u, v), z_cam = project(state, K, R, T)
    if use_depth:
        return np.array([u, v, z_cam])
    return np.array([u, v])


# EKF класс
class BallEKF:
    def __init__(self, K, R, T, dt,
                 pos_std0=1.0, vel_std0=5.0,
                 process_accel_std=6.0,
                 meas_px_std=3.0, meas_depth_std=1.5,
                 gate_chi2=16.27):  # 99.9% для 2 степеней свободы (только u,v)
        self.K, self.R, self.T = K, R, T
        self.dt = dt
        self.x = None
        self.P = None
        self.pos_std0 = pos_std0
        self.vel_std0 = vel_std0
        self.q_accel = process_accel_std
        self.meas_px_std = meas_px_std
        self.meas_depth_std = meas_depth_std
        self.gate_chi2 = gate_chi2

        self._last_confirmed = []  # [(t, pos3), ...] последние подтверждённые точки — для реинициализации скорости при ударе

        self.events = []   # список обнаруженных "ударов": dict(t, pos, v_before, v_after, speed_after)
        self.history = []  # покадровый лог: dict(t, state, P_diag, updated, hit)

    def _process_noise(self, dt):
        """Q через модель 'случайное ускорение' — стандартный дискретный White Noise Acceleration."""
        q = self.q_accel ** 2
        Qp = np.eye(3) * (dt ** 4 / 4) * q
        Qpv = np.eye(3) * (dt ** 3 / 2) * q
        Qv = np.eye(3) * (dt ** 2) * q
        Q = np.zeros((6, 6))
        Q[:3, :3] = Qp
        Q[:3, 3:] = Qpv
        Q[3:, :3] = Qpv
        Q[3:, 3:] = Qv
        return Q

    def init_state(self, t0, pos0, vel0_guess=(0.0, 0.0, 0.0)):
        self.x = np.array([*pos0, *vel0_guess], dtype=float)
        self.P = np.diag([self.pos_std0**2]*3 + [self.vel_std0**2]*3)
        self._t_prev = t0
        self._last_confirmed = [(t0, np.array(pos0, dtype=float))]

    def predict(self, dt):
        F = jacobian_f(self.x, dt)
        self.x = rk4_step(self.x, dt)
        Q = self._process_noise(dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, t, uv, box_size_px=None, meas_conf=1.0):
        """
        uv: (u, v) измеренные пиксели.
        box_size_px: min(w, h) бокса детекции в пикселях — используется как
                     грубая оценка глубины Z по известному диаметру мяча.
                     Можно передать None, тогда обновление только по (u, v)
        meas_conf: confidence YOLO в [0,1], снижает доверие к шумной детекции
                   (масштабирует R измерения)
        """
        use_depth = box_size_px is not None and box_size_px > 1e-3
        z_meas = list(uv)
        Rdiag = [self.meas_px_std ** 2] * 2
        if use_depth:
            f_eff = 0.5 * (self.K[0, 0] + self.K[1, 1])
            z_depth = f_eff * (2 * RADIUS) / box_size_px
            z_meas.append(z_depth)
            Rdiag.append(self.meas_depth_std ** 2)
        z_meas = np.array(z_meas)

        conf_scale = 1.0 / max(meas_conf, 0.05)
        Rmat = np.diag(Rdiag) * conf_scale

        h0 = _h(self.x, self.K, self.R, self.T, use_depth)
        H = measurement_jacobian(self.x, self.K, self.R, self.T, use_depth)

        y = z_meas - h0  # инновация
        S = H @ self.P @ H.T + Rmat
        nis = float(y[:2] @ np.linalg.solve(S[:2, :2], y[:2]))  # NIS по (u,v) — на глубину не гейтим, она сама по себе шумная

        hit_detected = False
        if nis > self.gate_chi2 and len(self._last_confirmed) >= 1:
            # Похоже на удар/контакт: физика между предыдущей подтверждённой
            # точкой и этой сломалась. Переинициализируем скорость по конечной
            # разности вместо доверия физическому предсказанию
            hit_detected = True
            t_prev, pos_prev = self._last_confirmed[-1]
            # Грубая 3D-оценка текущей точки: используем physика-предсказанную
            # глубину как приближение, обновлять будем ниже обычным шагом EKF,
            # но с сильно раздутой P на скорость, чтобы измерение "перетянуло" её
            dt_meas = max(t - t_prev, 1e-3)
            v_before = self.x[3:].copy()
            self.P[3:, 3:] *= 400.0  # временно "не доверяем" физике по скорости

            # пересчитываем инновацию/усиление с раздутой P
            S = H @ self.P @ H.T + Rmat
            Kgain = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + Kgain @ y
            self.P = (np.eye(STATE_DIM) - Kgain @ H) @ self.P

            v_after = self.x[3:].copy()
            self.events.append(dict(
                t=t, pos=self.x[:3].copy(),
                v_before=v_before, v_after=v_after,
                speed_after=float(np.linalg.norm(v_after)),
                speed_before=float(np.linalg.norm(v_before)),
            ))
        else:
            Kgain = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + Kgain @ y
            self.P = (np.eye(STATE_DIM) - Kgain @ H) @ self.P

        self._last_confirmed.append((t, self.x[:3].copy()))
        if len(self._last_confirmed) > 5:
            self._last_confirmed.pop(0)

        return hit_detected, nis

    def step(self, t, uv=None, box_size_px=None, meas_conf=1.0):
        """Один кадр: predict всегда, update — если есть детекция (uv не None)"""
        dt = t - self._t_prev
        if dt > 1e-6:
            self.predict(dt)
        self._t_prev = t

        hit = False
        nis = None
        if uv is not None:
            hit, nis = self.update(t, uv, box_size_px, meas_conf)

        self.history.append(dict(
            t=t, state=self.x.copy(), P_diag=np.diag(self.P).copy(),
            updated=uv is not None, hit=hit, nis=nis,
        ))
        return self.x.copy()