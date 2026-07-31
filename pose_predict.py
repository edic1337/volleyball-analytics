import cv2
from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')

video_path = 'match.mp4'
cap = cv2.VideoCapture(video_path)

# Получаем параметры исходного видео для сохранения результата
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output_pose_tracked.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      fps, (width, height))

print("Начинаем обработку видео...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Делаем детекцию и трекинг одновременно
    # persist=True говорит трекеру запоминать ID с прошлых кадров
    # tracker="bytetrack.yaml" включает продвинутый алгоритм отслеживания
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # Извлекаем данные для аналитики
    if results[0].boxes.id is not None:
        # Получаем ID всех игроков на кадре (например: [1, 2, 3, 4, 5, 6])
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Получаем координаты ключевых точек (17 точек на каждого человека)
        # Формат: [количество_людей, 17_суставов, 3_значения(x, y, confidence)]
        keypoints = results[0].keypoints.data.cpu().numpy()

        for i, track_id in enumerate(track_ids):
            player_keypoints = keypoints[i]
            
            # Пример: Достаем координаты правого и левого запястья (индексы 9 и 10 в COCO)
            left_wrist = player_keypoints[9]
            right_wrist = player_keypoints[10]
            
            # Если нейросеть уверена в точке (confidence > 0.5)
            if left_wrist[2] > 0.5:
                # Здесь в будущем будет логика: 
                # if math.dist(left_wrist_xy, ball_xy) < threshold: print(f"Игрок {track_id} коснулся мяча!")
                pass

    # Отрисовываем результаты на кадре (Скелеты + ID)
    annotated_frame = results[0].plot()
    
    # Записываем кадр в новый файл
    out.write(annotated_frame)

cap.release()
out.release()
print("Готово! Видео сохранено")