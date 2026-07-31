import os
import sys

# Папка с кадрами
FOLDER = "dataset_raw_frames"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    if not os.path.isdir(FOLDER):
        print(f"Папка '{FOLDER}' не найдена рядом со скриптом.")
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(FOLDER)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    if not files:
        print("Файлы изображений не найдены.")
        return

    deleted = 0
    kept = []
    for i, filename in enumerate(files, start=1): 
        if i % 4 == 0:
            path = os.path.join(FOLDER, filename)
            os.remove(path)
            deleted += 1
            print(f"Удалён: {filename}")
        else:
            kept.append(filename)

    print(f"\nУдалено файлов: {deleted}. Осталось: {len(kept)}")

    tmp_names = []
    for i, filename in enumerate(kept):
        ext = os.path.splitext(filename)[1].lower()
        src = os.path.join(FOLDER, filename)
        tmp = os.path.join(FOLDER, f"__tmp_{i}{ext}")
        os.rename(src, tmp)
        tmp_names.append((tmp, ext))

    for i, (tmp, ext) in enumerate(tmp_names):
        final_name = f"frame_{i}{ext}"
        final_path = os.path.join(FOLDER, final_name)
        os.rename(tmp, final_path)
        print(f"{tmp} -> {final_name}")

    print(f"\nГотово. Итоговое число файлов: {len(tmp_names)}")


if __name__ == "__main__":
    main()