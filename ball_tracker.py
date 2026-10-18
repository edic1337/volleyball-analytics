import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = "trimmed_clip_short.mp4"
YOLO_WEIGHTS = "runs/detect/train-18/weights/best.pt"
#YOLO_WEIGHTS = "yolo11s.pt"  # аххаах она просто все видео типа отслеживала 
CONF_THRESHOLD = 0.26
IOU_THRESHOLD = 0.45
IMG_SIZE = 1280 
OUTPUT_VIDEO = "tracked_output_1.mp4"
OUTPUT_TRAJECTORY = "ball_trajectory_1.npy"

def main():
    model = YOLO(YOLO_WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time = 1.0 / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    trajectory = []
    current_time = 0.0
    frame_idx = 0
    found_count = 0
    
    last_center = None
    MAX_PIXEL_JUMP = 90.0  # Максимальное смещение центра мяча за 1 кадр
    frames_lost = 0
    MAX_LOST_FRAMES = 12

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )[0]

        # Проверяем, нашел ли YOLO хоть какие-то объекты 
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            best_idx = -1
            
            if last_center is None:
                # Захват цели на первом успешном кадре: берем самую уверенную детекцию
                best_idx = confs.argmax()
            else:
                # Цель уже была, ищем ближайшую рамку к прошлой позиции
                min_dist = float('inf')
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    dist = np.hypot(cx - last_center[0], cy - last_center[1])
                    
                    if dist < min_dist and dist < MAX_PIXEL_JUMP:
                        min_dist = dist
                        best_idx = i

            if best_idx != -1:
                x1, y1, x2, y2 = boxes[best_idx].astype(int)
                conf = confs[best_idx]
                w = x2 - x1
                h = y2 - y1
                box_size = min(w, h)
                
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                last_center = (center_x, center_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"P {conf:.2f}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                trajectory.append([current_time, float(center_x), float(center_y), float(box_size)])
                found_count += 1
            else:
                # В заданном радиусе нужного мяча нет
                frames_lost += 1
                
                if frames_lost > MAX_LOST_FRAMES:
                    # Ждали слишком долго — забываем старую позицию
                    last_center = None
                    cv2.putText(frame, "TARGET RESET. SEARCHING...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    cv2.putText(frame, "WAITING NEAR LAST POS", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # YOLO вообще ничего не увидел — это тоже потеря цели, а не
            # отдельный случай. Раньше frames_lost здесь не увеличивался,
            # и таймаут MAX_LOST_FRAMES никогда не срабатывал при полном
            # пропадании мяча (самый частый случай: улетел за кадр, закрыт
            # игроком, смазан до неузнаваемости).
            frames_lost += 1
            if frames_lost > MAX_LOST_FRAMES:
                last_center = None
            cv2.putText(frame, "BALL LOST", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        writer.write(frame)

        frame_idx += 1
        current_time += frame_time
        if frame_idx % 100 == 0:
            print(f"  обработано кадров: {frame_idx}")


    cap.release()
    writer.release()

    traj_arr = np.array(trajectory)
    np.save(OUTPUT_TRAJECTORY, traj_arr)

    print(f"\nГотово. Всего кадров: {frame_idx}, мяч найден на {found_count} из них "
          f"({100 * found_count / max(frame_idx, 1):.1f}%).")
    print(f"Видео с рамками сохранено: {OUTPUT_VIDEO}")
    print(f"Координаты (t, u, v) сохранены: {OUTPUT_TRAJECTORY}")

if __name__ == "__main__":
    main()