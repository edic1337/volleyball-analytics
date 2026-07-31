import cv2
import os
from pathlib import Path

def extract_frames_for_cvat(video_path, output_folder="dataset_raw_frames", frame_step=3):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    # Вытаскиваем имя файла без расширения 
    video_name = Path(video_path).stem
    
    frame_count = 0
    saved_count = 0

    print(f"Начинаем нарезку видео: {video_name}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_step == 0:
            # Формируем безопасное имя: префикс_номер.jpg
            filename = os.path.join(output_folder, f"{video_name}_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Готово! Извлечено {saved_count} кадров.")

if __name__ == "__main__":
    # Теперь для каждого нового ролика просто меняешь название входного файла
    extract_frames_for_cvat(r"D:\volley_stat\dataset_videos\Skysteps 1 сет.f137.mp4", frame_step=4)
    extract_frames_for_cvat(r"D:\volley_stat\dataset_videos\Игорный институт сет 5.f137.mp4", frame_step=4)
    extract_frames_for_cvat(r"D:\volley_stat\dataset_videos\Росмолекула 2 сет.f137.mp4", frame_step=4)
    extract_frames_for_cvat(r"D:\volley_stat\dataset_videos\Тень на плетень 1 сет.f137.mp4", frame_step=4)
    extract_frames_for_cvat(r"D:\volley_stat\trainds.mp4", frame_step=5)