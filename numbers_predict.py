from ultralytics import YOLO

def main():
    print("Инициализация модели YOLOv8 Nano (Classification)...")
    
    # Модель для классификации 
    model = YOLO('yolov8n-cls.pt')

    # Настраиваем и запускаем обучение
    results = model.train(
        data='dataset_numbers',  # Имя главной папки с датасетом
        epochs=40,               # Количество проходов по датасету
        imgsz=64,                # Сжимаем картинки до 64x64 
        device=0,                # Явно указываем использовать GPU
        batch=32,                # Размер пакета изображений 
        name='volley_numbers',   # Имя папки, куда сохранятся результаты эксперимента
        project='runs/classify'  # Главная директория для логов
    )

    print("Обучение успешно завершено!")

if __name__ == '__main__':
    main()