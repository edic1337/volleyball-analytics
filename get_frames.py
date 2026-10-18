import cv2
import subprocess
import os

def cut_video_ffmpeg(input_path, output_path, start_time, end_time):
    print(f"Вырезаем видео через FFmpeg: с {start_time} по {end_time}...")
    command = [
        "ffmpeg",
        "-ss", start_time,
        "-to", end_time,
        "-i", input_path,
        "-c", "copy",  # Копируем кодеки без пережатия (очень быстро)
        output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Успешно сохранено в {output_path}")
    except FileNotFoundError:
        print("FFmpeg не установлен. Используйте Вариант 2 (OpenCV).")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка FFmpeg: {e}")


def cut_video_opencv(input_path, output_path, start_sec, end_sec):
    print(f"Вырезаем видео через OpenCV: с {start_sec}с по {end_sec}с...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames = end_frame - start_frame

    # Перематываем на нужный кадр
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break

        writer.write(frame)
        frames_processed += 1

        if frames_processed % int(fps * 10) == 0:
            print(f"  Обраработано кадров: {frames_processed} из {total_frames}")

    cap.release()
    writer.release()
    print(f"Готово! Сохранено в {output_path}")


if __name__ == "__main__":
    INPUT_VIDEO = "26-02-28-practice.mp4"      
    OUTPUT_VIDEO = "trimmed_clip_short_2.mp4"  
    
    START_STR = "01:16:00"
    END_STR = "01:16:40"
    cut_video_ffmpeg(INPUT_VIDEO, OUTPUT_VIDEO, START_STR, END_STR)