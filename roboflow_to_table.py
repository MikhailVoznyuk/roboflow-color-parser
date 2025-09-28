# tools/roboflow_to_plants.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
import pandas as pd

# -------------------- утилиты I/O --------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_imencode_jpg(img: np.ndarray, quality: int = 90):
    """Безопасно кодирует JPEG. Возвращает bytes или None."""
    # если вдруг однослойное изображение — конвертируем в BGR
    tmp = img
    if img.ndim == 2:
        tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", tmp, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok or buf is None or buf.size == 0:
        return None
    return buf.tobytes()

def safe_write_image(path: Path, img: np.ndarray, quality: int = 90) -> bool:
    """Пишет JPEG 'железно': сначала через имкодек+write_bytes, потом fallback на imwrite.
    Возвращает True/False."""
    ensure_dir(path.parent)
    data = safe_imencode_jpg(img, quality=quality)
    if data is not None:
        try:
            path.write_bytes(data)
            return True
        except Exception:
            pass
    # Fallback: cv2.imwrite (если по какой-то причине write_bytes не сработал)
    try:
        return bool(cv2.imwrite(str(path), img))
    except Exception:
        return False

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """Возвращает кроп или None, если после обрезки размер < 2x2."""
    H, W = img.shape[:2]
    # защита от отрицательных ширин/высот
    w = max(0, int(w)); h = max(0, int(h))
    x = int(x); y = int(y)

    # обрезаем по границам кадра
    x0 = max(0, min(W-1, x))
    y0 = max(0, min(H-1, y))
    x1 = max(0, min(W, x + w))
    y1 = max(0, min(H, y + h))

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return img[y0:y1, x0:x1].copy()

def to_int_bbox(b):
    x, y, w, h = b
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))

# -------------------- рендер: безопасные подписи --------------------
def safe_box_for_text(W:int, H:int, tw:int, th:int, x:int, y:int, prefer_above=True, pad=4):
    """Подбираем окно под текст так, чтобы не вылезало за кадр."""
    if prefer_above:
        x0 = x
        y0 = y - th - 2*pad - 2
    else:
        x0 = x
        y0 = y + 2
    if y0 < 0:                        # не влезает сверху — ставим снизу
        y0 = min(H - th - 2*pad, y + 2)
    if y0 + th + 2*pad > H:           # не влезает снизу — наверх
        y0 = max(0, y - th - 2*pad - 2)
    if x0 + tw + 2*pad > W:
        x0 = max(0, W - tw - 2*pad)
    if x0 < 0:
        x0 = 0
    return int(x0), int(y0), int(x0 + tw + 2*pad), int(y0 + th + 2*pad)

def put_tag_multiline(img: np.ndarray, anchor_xy: Tuple[int,int], lines: List[str],
                      bg: Tuple[int,int,int], prefer_above=True, pad=4,
                      font_scale_init=0.62, thickness=1):
    """Многострочная плашка + чёрная окантовка текста для читаемости."""
    H, W = img.shape[:2]
    fs = float(font_scale_init)
    # измеряем ширину/высоту блока
    def measure(fs_):
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs_, thickness)[0] for t in lines]
        tw = max(w for (w,h) in sizes) if sizes else 0
        th = sum(h for (w,h) in sizes) + (len(lines)-1)*2
        return tw, th, sizes
    tw, th, sizes = measure(fs)
    while (tw + 2*pad > W) and fs > 0.34:
        fs -= 0.04
        tw, th, sizes = measure(fs)

    x, y = int(anchor_xy[0]), int(anchor_xy[1])
    x0, y0, x1, y1 = safe_box_for_text(W, H, tw, th, x, y, prefer_above, pad)

    # фон
    cv2.rectangle(img, (x0, y0), (x1, y1), bg, -1)
    # лейдер: тонкая линия от якоря к плашке
    px = max(x0, min(x, x1))
    py = max(y0, min(y, y1))
    cv2.line(img, (x, y), (px, py), bg, 1, cv2.LINE_AA)

    # рисуем строки с чёрной окантовкой
    yy = y0 + pad
    for (t, (tw_i, th_i)) in zip(lines, sizes):
        org = (x0 + pad, yy + th_i)
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), thickness, cv2.LINE_AA)
        yy += th_i + 2

def draw_poly(img, pts, color, alpha=0.28, thickness=2):
    overlay = img.copy()
    if len(pts) >= 3:
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
        img[:] = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, color, thickness, cv2.LINE_AA)

def polygon_anchor(polys: List[List[Tuple[int,int]]]) -> Tuple[int,int]:
    """Центроид самого большого полигона (fallback — первая точка)."""
    if not polys: return 0, 0
    cs = [np.array(p, np.int32) for p in polys if len(p) >= 3]
    if not cs: return polys[0][0]
    areas = [abs(cv2.contourArea(c)) for c in cs]
    k = int(np.argmax(areas))
    M = cv2.moments(cs[k])
    if M["m00"] > 1e-6:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        return cx, cy
    x,y,w,h = cv2.boundingRect(cs[k])
    return int(x), int(y)

# -------------------- COCO helpers --------------------
def parse_segmentation(seg) -> Tuple[str, List[List[Tuple[int,int]]], int]:
    if seg is None: return "none", [], 0
    if isinstance(seg, list):  # polygons
        polys = []
        for arr in seg:
            if not arr: continue
            pts = [(int(round(arr[i])), int(round(arr[i+1])))
                   for i in range(0, len(arr), 2)]
            if len(pts) >= 3: polys.append(pts)
        return "polygon", polys, 0
    if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:  # RLE
        counts = seg.get('counts')
        return "rle", [], (len(counts) if hasattr(counts, '__len__') else 0)
    return "none", [], 0

# Цвета классов
CLASS_COLORS = {
    "tree": (50, 180, 80),
    "shrub": (140, 70, 220),
    "fungus": (0, 150, 255),
    "cavity": (0, 0, 230),
    "crack": (50, 50, 50),
    "mech_damage": (0, 90, 200),
    "pests": (170, 90, 200),
    "__other__": (200, 200, 200),
}
PLANT_CLASSES = {"tree", "shrub"}

# -------------------- основной проход по одному split --------------------
def process_split(split_dir: Path, split_name: str,
                  out_viz_root: Path, out_crops_root: Path | None) -> List[Dict]:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[WARN] {split_name}: нет {ann_path}")
        return []

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    cats_by_id = {c["id"]: c for c in coco.get("categories", [])}

    anns_by_img: Dict[Any, List[Dict]] = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    out_rows: List[Dict] = []
    viz_split = out_viz_root / split_name
    ensure_dir(viz_split)
    crops_split = None
    if out_crops_root is not None:
        crops_split = out_crops_root / split_name
        ensure_dir(crops_split)

    for img_id, anns in anns_by_img.items():
        im_info = images_by_id.get(img_id)
        if im_info is None:
            print(f"[WARN] {split_name}: image_id={img_id} отсутствует в images")
            continue

        file_name = im_info["file_name"]
        W, H = int(im_info.get("width", 0)), int(im_info.get("height", 0))
        img_path = split_dir / file_name
        if not img_path.exists():
            alts = list(split_dir.rglob(Path(file_name).name))
            if alts: img_path = alts[0]
        if not img_path.exists():
            print(f"[WARN] нет файла изображения: {img_path}")
            continue

        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] не удалось открыть: {img_path}")
            continue

        # если в COCO высота/ширина нули — берём реальные из изображения
        if W <= 0 or H <= 0:
            H, W = img.shape[:2]

        canvas = img.copy()
        anns_sorted = sorted(anns, key=lambda a: int(a.get("id", 0)))

        for a in anns_sorted:
            ann_id = int(a.get("id", -1))
            if ann_id < 0:
                ann_id = int(abs(hash((img_id, a.get("category_id"), tuple(a.get("bbox", [])))))) % (10**9)

            cat = cats_by_id.get(a.get("category_id"), {"name":"__other__"})
            name = str(cat.get("name", "__other__"))
            color = CLASS_COLORS.get(name, CLASS_COLORS["__other__"])

            x, y, w_box, h_box = to_int_bbox(a.get("bbox", [0,0,0,0]))
            seg_type, polys, rle_len = parse_segmentation(a.get("segmentation"))

            # --- рисуем объект ---
            anchor = (x, y)  # по умолчанию — угол бокса
            if seg_type == "polygon" and polys:
                for pts in polys: draw_poly(canvas, pts, color, alpha=0.26, thickness=2)
                anchor = polygon_anchor(polys)
            else:
                cv2.rectangle(canvas, (x, y), (x + w_box, y + h_box), color, 2, cv2.LINE_AA)

            # маленькая точка-якорь внутри полигона/бокса
            cv2.circle(canvas, (int(anchor[0]), int(anchor[1])), 3, color, -1, cv2.LINE_AA)

            # подписи двумя строками — всегда читабельны
            tag_lines = [f"img:{img_id}", f"ann:{ann_id}  {name}"]
            put_tag_multiline(canvas, anchor, tag_lines, color, prefer_above=True)

            # --- только растения -> строка таблицы plants ---
            if name in PLANT_CLASSES:
                row = {
                    # ОБЯЗАТЕЛЬНАЯ СХЕМА (пустые поля для дендрологов)
                    "image_id": str(img_id),
                    "plant_id": str(ann_id),
                    "species": "",
                    "species_confidence": "",          # {1,2,3}
                    "fungus_severity": "",             # {0,1,2,3}
                    "cavity_severity": "",             # {0,1,2,3}
                    "crack_severity": "",              # {0,1,2,3}
                    "mech_damage_severity": "",        # {0,1,2,3}
                    "pests_severity": "",              # {0,1,2,3}
                    "plant_health": "",                # {0,1,2,3}
                    "leaves": "",                      # {0,1,2}
                    "month": "",                       # {0,1..12}
                    "tilt_deg": "",                    # целое -1 или 0..90

                    # ДОП. УДОБНЫЕ ПОЛЯ (можно игнорировать при обучении)
                    "plant_type": name,                # tree | shrub
                    "image_file": file_name,
                    "viz_path": str((viz_split / file_name).as_posix()),
                    "crop_path": "",                   # заполним ниже
                }
                # --- устойчивое извлечение кропа ---
                crop_saved = False
                if crops_split is not None:
                    _x, _y, _w, _h = x, y, w_box, h_box

                    # если bbox "битый", а есть полигон — берём bbox по полигону
                    if (_w <= 1 or _h <= 1) and seg_type == "polygon" and polys:
                        cnt = np.array(polys[0], dtype=np.int32)
                        rx, ry, rw, rh = cv2.boundingRect(cnt)
                        _x, _y, _w, _h = rx, ry, rw, rh

                    crop = safe_crop(img, _x, _y, _w, _h)
                    if crop is not None and crop.size > 0:
                        crop_name = f"{Path(file_name).stem}__img{img_id}_ann{ann_id}_{name}.jpg"
                        crop_path = crops_split / crop_name
                        ensure_dir(crops_split)
                        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        if ok:
                            buf.tofile(str(crop_path))
                            row["crop_path"] = str(crop_path.as_posix())
                            crop_saved = True

                    if not crop_saved:
                        # подробный лог, чтобы быстро понять проблемную аннотацию
                        print(f"[WARN] skip empty crop: file={file_name} img:{img_id} ann:{ann_id} "
                              f"bbox=({x},{y},{w_box},{h_box}) W={W} H={H} seg_type={seg_type}")

                out_rows.append(row)

        # сохраняем визуализацию кадра
        vp = viz_split / file_name
        ensure_dir(vp.parent)
        ok_viz = safe_write_image(vp, canvas, quality=90)
        if not ok_viz:
            print(f"[WARN] viz save failed: {vp}")

    return out_rows

# -------------------- CLI --------------------
def find_splits(root: Path) -> List[Tuple[str, Path]]:
    """Возвращает пары (split_name, split_path). Понимает и корень с train/valid/test, и одиночную папку."""
    splits: List[Tuple[str, Path]] = []
    for s in ("train", "valid", "test"):
        p = root / s
        if (p / "_annotations.coco.json").exists():
            splits.append((s, p))
    if splits:
        return splits
    # одиночный каталог?
    if (root / "_annotations.coco.json").exists():
        return [(root.name, root)]
    return []

def main():
    ap = argparse.ArgumentParser(description="COCO (Roboflow) -> plants.csv + читаемые визуализации с img/ann id.")
    ap.add_argument("--root", required=True, type=str, help="Корень экспорта Roboflow (или конкретный split).")
    ap.add_argument("--out",  required=True, type=str, help="Куда сохранить CSV и визуализации.")
    ap.add_argument("--no-crops", action="store_true", help="Не сохранять кропы растений.")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    viz_root = out / "viz"
    crops_root = None if args.no_crops else (out / "crops")

    splits = find_splits(root)
    if not splits:
        print("[ERR] Не найден(ы) _annotations.coco.json. Проверьте --root.")
        sys.exit(2)

    all_rows: List[Dict] = []
    for split_name, split_path in splits:
        rows = process_split(split_path, split_name, viz_root, crops_root)
        all_rows.extend(rows)
        print(f"[OK] {split_name}: plants={len(rows)}")

    ensure_dir(out)
    out_csv = out / "plants.csv"
    df = pd.DataFrame(all_rows, columns=[
        # строгий порядок «боевой» схемы
        "image_id","plant_id","species","species_confidence",
        "fungus_severity","cavity_severity","crack_severity","mech_damage_severity","pests_severity",
        "plant_health","leaves","month","tilt_deg",
        # доп. для удобства
        "plant_type","image_file","viz_path","crop_path",
    ])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] plants.csv -> {out_csv}")
    print(f"[INFO] viz -> {viz_root}")
    if crops_root: print(f"[INFO] crops -> {crops_root}")

if __name__ == "__main__":
    main()
