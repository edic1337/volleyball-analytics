import cv2
import math
import numpy as np
from ultralytics import YOLO

def main():
    # Настройки
    VIDEO_PATH = "match.mp4"
    YOLO_WEIGHTS = "yolov8n.pt"
    MAX_DISTANCE = 100 # Радиус доверия в пикселях (максимальное отклонение от прогноза)

    model = YOLO(YOLO_WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    current_time = 0.0

    # Массив для хранения отфильтрованной траектории: [t, u, v]
    trajectory = [] 

    print("Запуск трекера. 'q' для выхода.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Делаем детекцию всех объектов на кадре
        # conf=0.25 - снижаем порог, чтобы YOLO находила даже смазанный мяч
        results = model(frame, conf=0.25, verbose=False)
        
        # Собираем центры всех найденных объектов
        candidates = []
        if results[0].boxes.id is not None or len(results[0].boxes) > 0:
            for box in results[0].boxes.xywh: # Формат: x_center, y_center, width, height
                x, y, w, h = box.cpu().numpy()
                candidates.append((x, y))

        best_point = None

        # Логика голосования
        if len(trajectory) < 2:
            # У нас еще нет вектора скорости. 
            # На практике лучше всего первые 2 кадра кликнуть мышкой (как в калибровке).
            # Но для автоматики пока берем первую попавшуюся детекцию.
            if len(candidates) > 0:
                best_point = candidates[0]
        else:
            # Физика инерции
            # Берем координаты с двух предыдущих кадров
            _, u_prev, v_prev = trajectory[-1]
            _, u_prev2, v_prev2 = trajectory[-2]
            
            # Вычисляем 2D-вектор скорости (в пикселях за кадр)
            du = u_prev - u_prev2
            dv = v_prev - v_prev2
            
            # Ожидаемая точка мяча на текущем кадре
            u_pred = u_prev + du
            v_pred = v_prev + dv
            
            # ИЗМЕРЕНИЕ И КОРРЕКЦИЯ (Компьютерное зрение)
            min_dist = float('inf')
            
            for (u, v) in candidates:
                # Считаем Евклидово расстояние от прогноза до того, что видит YOLO
                dist = math.dist((u_pred, v_pred), (u, v))
                
                # Голосование: Если точка рядом с прогнозом, мы ей верим
                if dist < min_dist and dist < MAX_DISTANCE:
                    min_dist = dist
                    best_point = (u, v)

        if best_point:
            # Мяч найден и подтвержден. Сохраняем данные.
            trajectory.append([current_time, best_point[0], best_point[1]])
            
            # Рисуем зеленый кружок на самом мяче
            cv2.circle(frame, (int(best_point[0]), int(best_point[1])), 8, (0, 255, 0), -1)
            
            # Рисуем хвост (траекторию) для красоты
            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i-1][1]), int(trajectory[i-1][2]))
                pt2 = (int(trajectory[i][1]), int(trajectory[i][2]))
                cv2.line(frame, pt1, pt2, (0, 165, 255), 2)
        else:
            # YOLO потеряла мяч (перекрыли руками или он вылетел за кадр).
            # Можно оставить пропуск кадра, least_squares спокойно работает с пропусками по времени.
            cv2.putText(frame, "BALL LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Smart Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        current_time += frame_time

    cap.release()
    cv2.destroyAllWindows()

    # В конце сохраняем собранные данные для физического движка
    traj_arr = np.array(trajectory)
    np.save('ball_trajectory.npy', traj_arr)
    print(f"\nТрекинг завершен. Собрано {len(traj_arr)} точек. Сохранено в 'ball_trajectory.npy'")

if __name__ == "__main__":
    main()