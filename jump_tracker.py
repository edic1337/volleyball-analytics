"""
Подсчёт прыжков волейболистов (кол-во + высота) с атрибуцией по номеру на футболке.

Идея пайплайна:
    1. YOLOv8-pose + встроенный трекер (ByteTrack/BoT-SORT) дают за один вызов:
       bbox, персистентный track_id, ключевые точки скелета.
    2. Для каждого track_id по щиколоткам определяем момент "стоит на полу" (Z=0)
       и точку (X, Y) на площадке через обратную проекцию (P-матрица, Z=0 -> плоскость пола).
    3. Пока (X, Y) считается фиксированной (игрок не сместился заметно по горизонтали),
       вертикальное положение таза переводится в метры вдоль этого луча -> Z(t).
    4. Фаза "в воздухе" (Z(t) выше базовой линии) фиттится параболой с g = 9.81 -> высота, hang time.
    5. Номер на футболке читается НЕ каждый кадр, а по расписанию (раз в N кадров на трек,
       либо когда бокс крупный/трек только что появился) -> голосование по накопленным чтениям
       -> присвоение roster_id треку. Статистика по треку копится всегда, даже до того,
       как номер прочитан, и переносится в профиль игрока постфактум.

Внешние зависимости:
    pip install ultralytics opencv-python numpy scipy easyocr
"""

from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from scipy.optimize import least_squares

class CameraModel:
    """Обёртка над калибровочной матрицей P
    """

    def __init__(self, P: np.ndarray):
        assert P.shape == (3, 4), f"Ожидалась матрица 3x4, получено {P.shape}"
        self.P = P

    @classmethod
    def from_file(cls, path: str) -> "CameraModel":
        path = Path(path)
        if path.suffix == ".npy":
            P = np.load(path)
        elif path.suffix == ".json":
            with open(path) as f:
                P = np.array(json.load(f), dtype=np.float64)
        else:
            raise ValueError("Ожидается .npy или .json с матрицей P (3x4)")
        return cls(P)

    def project(self, X: float, Y: float, Z: float) -> tuple[float, float]:
        """3D (метры) -> 2D пиксели."""
        p = self.P @ np.array([X, Y, Z, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def backproject_ground(self, u: float, v: float) -> tuple[float, float]:
        """Пиксель -> (X, Y) на плоскости пола (Z = 0).
        При Z=0 остаются три неизвестных: X, Y, s (масштаб однородных координат).
        Три уравнения проекции дают точную (не переопределённую) линейную систему.
        """
        P = self.P
        A = np.array([
            [P[0, 0], P[0, 1], -u],
            [P[1, 0], P[1, 1], -v],
            [P[2, 0], P[2, 1], -1.0],
        ])
        b = np.array([-P[0, 3], -P[1, 3], -P[2, 3]])
        X, Y, _s = np.linalg.solve(A, b)
        return float(X), float(Y)

    def solve_Z_at_fixed_XY(self, X: float, Y: float, u: float, v: float) -> float:
        """Пиксель + известные (X, Y) -> Z вдоль этого конкретного луча.

        Теперь неизвестна только одна пространственная координата (Z) плюс
        масштаб s -> система переопределена (2 уравнения на 1 неизвестную Z
        после исключения s), решаем методом наименьших квадратов.
        """
        P = self.P
        rhs_common = P[2, 0] * X + P[2, 1] * Y + P[2, 3]  # = s, если Z известен; здесь как коэфф.

        # Из строки 3: s = P20*X + P21*Y + P22*Z + P23
        # Подставляем в строки 1 и 2, получаем линейные уравнения относительно Z.
        A = np.array([
            [P[0, 2] - u * P[2, 2]],
            [P[1, 2] - v * P[2, 2]],
        ])
        b = np.array([
            u * (P[2, 0] * X + P[2, 1] * Y + P[2, 3]) - (P[0, 0] * X + P[0, 1] * Y + P[0, 3]),
            v * (P[2, 0] * X + P[2, 1] * Y + P[2, 3]) - (P[1, 0] * X + P[1, 1] * Y + P[1, 3]),
        ])
        Z, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(Z[0])



COCO_L_HIP, COCO_R_HIP = 11, 12
COCO_L_ANKLE, COCO_R_ANKLE = 15, 16
KPT_CONF_THRESHOLD = 0.4  # порог уверенности модели для детекции тела

# Пороги, которые почти наверняка придётся подстроить под вашу конкретную камеру/зал
GROUND_Z_TOLERANCE_M = 0.06      # если щиколотка в пределах +- этого значения от Z=0, то игрок на полу, значение в метрах
FLIGHT_Z_THRESHOLD_M = 0.10      # если таз выше этого значения, то игрок в фазе прыжка
MIN_FLIGHT_FRAMES = 4            # минимум кадров в фазе полёта, чтобы восстановить параболу прыжка и оченить высоту 
MAX_FIT_RMSE_M = 0.06            # если ошибка наложения траектории больше этого значения, то отбрасывем взятые кадры 
MIN_JUMP_HEIGHT_M = 0.10         # минимальная высота прыжка, остальное не учитываем 
G = 9.81

JERSEY_READ_EVERY_N_FRAMES = 20   # не читаем номер каждый кадр
JERSEY_MIN_BBOX_HEIGHT_PX = 65    # слишком мелкий бокс - смысла читать номер нет
JERSEY_VOTE_THRESHOLD = 0.55      # порог консенсуса для голосования, для окончательного присвоения номера конкретному треку, выбранная цифра должна набрать более 55% голосов среди всех уверенных попыток распознавания


@dataclass
class JumpEvent:
    track_id: int
    t_takeoff: float
    t_landing: float
    peak_height_m: float
    hang_time_s: float
    fit_rmse_m: float


@dataclass
class JerseyReading:
    frame_idx: int
    number: str
    confidence: float


@dataclass
class TrackState:
    track_id: int
    roster_id: Optional[str] = None

    # геометрия / базовая линия
    ground_XY: Optional[tuple[float, float]] = None
    hip_baseline_history: deque = field(default_factory=lambda: deque(maxlen=30))
    hip_baseline_z: Optional[float] = None

    # временной ряд высоты таза для детекции полёта
    height_buffer: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, Z)

    # состояние конечного автомата прыжка
    in_flight: bool = False
    flight_samples: list = field(default_factory=list)  # (t, Z) во время текущего полёта
    frames_since_last_jersey_read: int = 0

    jersey_readings: list[JerseyReading] = field(default_factory=list)
    jump_events: list[JumpEvent] = field(default_factory=list)

    last_seen_frame: int = 0


@dataclass
class PlayerProfile:
    roster_id: str
    merged_track_ids: list[int] = field(default_factory=list)
    jump_events: list[JumpEvent] = field(default_factory=list)

    @property
    def total_jumps(self) -> int:
        return len(self.jump_events)

    @property
    def avg_height_m(self) -> float:
        if not self.jump_events:
            return 0.0
        return float(np.mean([j.peak_height_m for j in self.jump_events]))

    @property
    def max_height_m(self) -> float:
        if not self.jump_events:
            return 0.0
        return float(np.max([j.peak_height_m for j in self.jump_events]))



def fit_flight_parabola(samples: list[tuple[float, float]]) -> Optional[dict]:
    """samples: список (t, Z) во время предполагаемого полёта.

    Возвращает параметры Z0, Vz0, RMSE и производные (высота, hang time),
    либо None, если данных недостаточно.
    """
    if len(samples) < MIN_FLIGHT_FRAMES:
        return None

    t = np.array([s[0] for s in samples])
    z = np.array([s[1] for s in samples])
    t0 = t[0]
    t_rel = t - t0

    def residuals(params):
        Z0, Vz0 = params
        z_pred = Z0 + Vz0 * t_rel - 0.5 * G * t_rel ** 2
        return z_pred - z

    # начальное приближение: Z0 = первое наблюдение, Vz0 из грубой разности
    Vz0_guess = (z[len(z) // 2] - z[0]) / max(t_rel[len(z) // 2], 1e-3)
    result = least_squares(residuals, x0=[z[0], Vz0_guess], loss="huber", f_scale=0.03)

    Z0, Vz0 = result.x
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))

    if Vz0 <= 0:
        # нет фазы взлёта в данных (например, поймали только нисходящую часть) -
        # пик всё равно можно оценить, но hang time будет недооценён; помечаем через rmse-фильтр ниже
        pass

    peak_extra_height = (Vz0 ** 2) / (2 * G) if Vz0 > 0 else 0.0
    peak_height = (Z0 - min(z.min(), Z0)) + peak_extra_height  # относительно нижней точки полёта
    hang_time = (2 * Vz0 / G) if Vz0 > 0 else (t_rel[-1] - t_rel[0])

    return {
        "Z0": float(Z0),
        "Vz0": float(Vz0),
        "rmse": rmse,
        "peak_height_m": float(max(peak_height, z.max() - z.min())),
        "hang_time_s": float(hang_time),
        "t0": float(t0),
        "t_end": float(t[-1]),
    }



class JerseyReader:
    """Тонкая обёртка над easyocr, ограниченная цифрами.
    """

    def __init__(self, use_gpu: bool = True):
        import easyocr  # ленивый импорт
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)

    def read_number(self, torso_crop: np.ndarray) -> Optional[tuple[str, float]]:
        if torso_crop.size == 0:
            return None
        # Лёгкий апскейл — при низком нативном разрешении номера это ощутимо помогает
        h, w = torso_crop.shape[:2]
        if h < 80:
            scale = 80 / h
            torso_crop = cv2.resize(torso_crop, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)

        results = self.reader.readtext(
            torso_crop,
            allowlist="0123456789",
            detail=1,
        )
        if not results:
            return None
        # берём самое уверенное распознанное число на кропе
        results.sort(key=lambda r: r[2], reverse=True)
        _, text, conf = results[0]
        text = text.strip()
        if not text.isdigit():
            return None
        return text, float(conf)


def crop_torso(frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Верхняя половина bbox игрока — зона номера на груди/спине."""
    x1, y1, x2, y2 = bbox_xyxy
    h = y2 - y1
    y_torso_end = y1 + int(h * 0.55)
    return frame[max(y1, 0):max(y_torso_end, 0), max(x1, 0):max(x2, 0)]


def should_read_jersey(track: TrackState, bbox_h_px: float) -> bool:
    if bbox_h_px < JERSEY_MIN_BBOX_HEIGHT_PX:
        return False
    if track.roster_id is not None:
        # уже определён — читаем изредка для доп. подтверждения, но не приоритетно
        return track.frames_since_last_jersey_read >= JERSEY_READ_EVERY_N_FRAMES * 4
    return track.frames_since_last_jersey_read >= JERSEY_READ_EVERY_N_FRAMES


def resolve_jersey_vote(readings: list[JerseyReading]) -> tuple[Optional[str], float]:
    strong = [r for r in readings if r.confidence > 0.5]
    if not strong:
        return None, 0.0
    counts = Counter(r.number for r in strong)
    number, votes = counts.most_common(1)[0]
    return number, votes / len(strong)



class TrackManager:
    def __init__(self, camera: CameraModel):
        self.camera = camera
        self.tracks: dict[int, TrackState] = {}
        self.profiles: dict[str, PlayerProfile] = {}

    def get_or_create(self, track_id: int, frame_idx: int) -> TrackState:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id=track_id)
        t = self.tracks[track_id]
        t.last_seen_frame = frame_idx
        return t

    def update_geometry(self, track: TrackState, keypoints: np.ndarray, t_sec: float):
        """keypoints: массив (17, 3) формата YOLOv8-pose: x, y, conf."""
        l_ankle, r_ankle = keypoints[COCO_L_ANKLE], keypoints[COCO_R_ANKLE]
        l_hip, r_hip = keypoints[COCO_L_HIP], keypoints[COCO_R_HIP]

        ankle_pts = [p for p in (l_ankle, r_ankle) if p[2] > KPT_CONF_THRESHOLD]
        hip_pts = [p for p in (l_hip, r_hip) if p[2] > KPT_CONF_THRESHOLD]
        if not ankle_pts or not hip_pts:
            return  # недостаточно уверенных keypoints в этом кадре — пропускаем

        ankle_u = float(np.mean([p[0] for p in ankle_pts]))
        ankle_v = float(np.mean([p[1] for p in ankle_pts]))
        hip_u = float(np.mean([p[0] for p in hip_pts]))
        hip_v = float(np.mean([p[1] for p in hip_pts]))

        # Оцениваем Z щиколотки, ПРЕДПОЛАГАЯ на секунду ту же (X,Y), что при последней
        # известной точке стояния — если её ещё нет, инициализируем через Z=0.
        if track.ground_XY is None:
            X, Y = self.camera.backproject_ground(ankle_u, ankle_v)
            track.ground_XY = (X, Y)

        X, Y = track.ground_XY
        ankle_Z = self.camera.solve_Z_at_fixed_XY(X, Y, ankle_u, ankle_v)
        hip_Z = self.camera.solve_Z_at_fixed_XY(X, Y, hip_u, hip_v)

        grounded = abs(ankle_Z) < GROUND_Z_TOLERANCE_M
        if grounded:
            # обновляем точку стояния и базовую линию таза, пока игрок точно на полу
            track.ground_XY = self.camera.backproject_ground(ankle_u, ankle_v)
            track.hip_baseline_history.append(hip_Z)
            track.hip_baseline_z = float(np.median(track.hip_baseline_history))

        track.height_buffer.append((t_sec, hip_Z))

        if track.hip_baseline_z is None:
            return  # ещё не видели игрока стабильно стоящим - не можем судить о прыжке

        elevation = hip_Z - track.hip_baseline_z

        # --- конечный автомат фазы полёта ---
        if not track.in_flight and elevation > FLIGHT_Z_THRESHOLD_M:
            track.in_flight = True
            track.flight_samples = [(t_sec, hip_Z)]
        elif track.in_flight and elevation > FLIGHT_Z_THRESHOLD_M * 0.5:
            track.flight_samples.append((t_sec, hip_Z))
        elif track.in_flight:
            # приземлились (elevation упала ниже половины порога) -> закрываем фазу и фиттим
            track.flight_samples.append((t_sec, hip_Z))
            self._finalize_jump(track)
            track.in_flight = False
            track.flight_samples = []

    def _finalize_jump(self, track: TrackState):
        fit = fit_flight_parabola(track.flight_samples)
        if fit is None:
            return
        if fit["rmse"] > MAX_FIT_RMSE_M:
            return  # физически невалидная форма — скорее всего шум трекинга/keypoints, отбрасываем
        if fit["peak_height_m"] < MIN_JUMP_HEIGHT_M:
            return  # не прыжок, а шум/лёгкое приседание

        event = JumpEvent(
            track_id=track.track_id,
            t_takeoff=fit["t0"],
            t_landing=fit["t_end"],
            peak_height_m=fit["peak_height_m"],
            hang_time_s=fit["hang_time_s"],
            fit_rmse_m=fit["rmse"],
        )
        track.jump_events.append(event)
        if track.roster_id is not None:
            self.profiles[track.roster_id].jump_events.append(event)

    def maybe_read_jersey(self, track: TrackState, frame: np.ndarray,
                           bbox_xyxy: tuple[int, int, int, int], frame_idx: int,
                           jersey_reader: "JerseyReader"):
        track.frames_since_last_jersey_read += 1
        bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
        if not should_read_jersey(track, bbox_h):
            return

        track.frames_since_last_jersey_read = 0
        crop = crop_torso(frame, bbox_xyxy)
        result = jersey_reader.read_number(crop)
        if result is None:
            return

        number, conf = result
        track.jersey_readings.append(JerseyReading(frame_idx, number, conf))

        if track.roster_id is None:
            roster_id, agreement = resolve_jersey_vote(track.jersey_readings)
            if roster_id is not None and agreement >= JERSEY_VOTE_THRESHOLD:
                self._assign_roster_id(track, roster_id)

    def _assign_roster_id(self, track: TrackState, roster_id: str):
        track.roster_id = roster_id
        profile = self.profiles.setdefault(
            roster_id, PlayerProfile(roster_id=roster_id)
        )
        profile.merged_track_ids.append(track.track_id)
        # ретроактивно переносим уже накопленную (анонимную) статистику по прыжкам
        profile.jump_events.extend(track.jump_events)

    def export_summary(self) -> dict:
        summary = {}
        for roster_id, profile in self.profiles.items():
            summary[roster_id] = {
                "total_jumps": profile.total_jumps,
                "avg_height_m": round(profile.avg_height_m, 3),
                "max_height_m": round(profile.max_height_m, 3),
                "merged_track_ids": profile.merged_track_ids,
            }
        unattributed = [
            t for t in self.tracks.values()
            if t.roster_id is None and t.jump_events
        ]
        if unattributed:
            summary["_unattributed"] = {
                f"track_{t.track_id}": {
                    "total_jumps": len(t.jump_events),
                    "avg_height_m": round(float(np.mean([j.peak_height_m for j in t.jump_events])), 3),
                }
                for t in unattributed
            }
        return summary

# Главный цикл

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Путь к видео")
    parser.add_argument("--calib", required=True, help="Путь к P-матрице (.npy или .json)")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt",
                         help="Веса YOLO pose-модели (можно свою дообученную)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Конфиг трекера ultralytics (bytetrack.yaml / botsort.yaml)")
    parser.add_argument("--out", default="jump_stats.json", help="Куда сохранить итоговую статистику")
    parser.add_argument("--no-gpu", action="store_true", help="Отключить GPU для OCR")
    parser.add_argument("--show", action="store_true", help="Показывать видео с разметкой")
    args = parser.parse_args()

    from ultralytics import YOLO  # ленивый импорт тяжёлой зависимости

    camera = CameraModel.from_file(args.calib)
    manager = TrackManager(camera)
    jersey_reader = JerseyReader(use_gpu=not args.no_gpu)

    pose_model = YOLO(args.pose_model)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = frame_idx / fps

        # YOLOv8-pose с встроенным трекером: bbox + track_id + keypoints одним вызовом
        results = pose_model.track(
            frame, persist=True, tracker=args.tracker, classes=[0], verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            all_keypoints = results.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for bbox, tid, kpts in zip(boxes_xyxy, track_ids, all_keypoints):
                bbox_xyxy = tuple(int(v) for v in bbox)
                track = manager.get_or_create(tid, frame_idx)

                manager.update_geometry(track, kpts, t_sec)
                manager.maybe_read_jersey(track, frame, bbox_xyxy, frame_idx, jersey_reader)

                if args.show:
                    x1, y1, x2, y2 = bbox_xyxy
                    label = track.roster_id or f"id{tid}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, f"#{label} jumps={len(track.jump_events)}",
                                (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if args.show:
            cv2.imshow("jump_tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    summary = manager.export_summary()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()