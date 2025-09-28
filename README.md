Инструменты для преобразования экспортов Roboflow в табличный вид и генерации визуализаций/кропов.

## Структура
```text
.gitignore
requirements.txt
roboflow_to_table.py
```

## Зависимости
```text
numpy==2.0.2
opencv-python==4.12.0.88
pandas==2.3.2
python-dateutil==2.9.0.post0
pytz==2025.2
six==1.17.0
tzdata==2025.2
```

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Использование
Пример запуска:
```bash
python roboflow_to_table.py --help
# типовой
python roboflow_to_table.py --root /absolute_path/to/roboflow/export --out /absolute_path/to/result/dir --no-crops
```

## Выходные данные
- CSV/Excel со сводной информацией
- Папка viz/ с изображениями и боксами
- Папка crops/ с вырезками объектов

## Примечания
- Убедись, что выходные директории существуют перед записью.
- Если ловишь `OSError: requested and 0 written` при `cv2.imencode(...).tofile(...)`, проверь путь, права и свободное место. На Windows по возможности избегай кириллицы в путях.
