import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = "YOLO_result.mp4"
YOLO_WEIGHTS = "runs/detect/train-18/weights/best.pt"
CONF_THRESHOLD = 0.25
OUTPUT_VIDEO = "tracked_output.mp4"
OUTPUT_TRAJECTORY = "ball_trajectory.npy"

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

    trajectory = []  # [t, u, v] только для кадров, где мяч реально найден
    current_time = 0.0
    frame_idx = 0
    found_count = 0

    print("Обработка видео...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        if len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            x, y, w, h = results.boxes.xywh[best_idx].cpu().numpy()
            conf = confs[best_idx]

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            trajectory.append([current_time, float(x), float(y)])
            found_count += 1
        else:
            cv2.putText(frame, "BALL LOST", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        writer.write(frame)

        frame_idx += 1
        current_time += frame_time
        if frame_idx % 200 == 0:
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