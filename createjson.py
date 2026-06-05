#Generate JSON Annotations

import os
import json
import cv2

# ===== 配置路径 =====
val_data_root = r"D:\CJJ\desk\SDM-Car-SU\U3\val_data"  # 例如包含 001、002、003...
output_json = r"D:\CJJ\desk\SDM-Car-SU\U3\annotations\val.json"  # 最终输出的 JSON 文件路径


# ===== 类别定义 =====
category = {
    "id": 1,
    "name": "car",
    "supercategory": "car"
}

# ===== COCO结构容器 =====
coco_output = {
    "images": [],
    "categories": [category],
    "annotations": [],
}

# ===== 全局变量 =====
image_id = 0
ann_id = 0

# 遍历 val_data 下的每个子文件夹
for seq_folder in sorted(os.listdir(val_data_root)):
    seq_path = os.path.join(val_data_root, seq_folder)
    img_dir = os.path.join(seq_path, "img1")
    label_path = os.path.join(seq_path, "gt.txt")

    if not os.path.exists(img_dir) or not os.path.isfile(label_path):
        continue  # 跳过无效文件夹

    # 收集图像文件名
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    frame_to_filename = {i+1: f for i, f in enumerate(image_files)}  # 从第1帧开始

    # 映射 frame_id -> image_id
    frame_to_image_id = {}

    # === 构建 images 字段 ===
    for frame_id, filename in frame_to_filename.items():
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]

        # 修改后的图像名，如 001_000001.jpg
        new_filename = f"{seq_folder}_{filename}"
        coco_output["images"].append({
            "file_name": new_filename,
            "height": height,
            "width": width,
            "id": image_id
        })
        frame_to_image_id[frame_id] = image_id
        image_id += 1

    # === 构建 annotations 字段 ===
    with open(label_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            if len(items) < 6:
                continue
            frame_id = int(items[0])
            obj_id = int(items[1])
            x = float(items[2])
            y = float(items[3])
            w = float(items[4])
            h = float(items[5])

            if frame_id not in frame_to_image_id:
                continue

            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": frame_to_image_id[frame_id],
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                "obj_id": str(obj_id)
            })
            ann_id += 1

# 写入 JSON 文件
with open(output_json, 'w') as f:
    json.dump(coco_output, f, indent=4)

print(f"✅ 转换完成，共包含 {image_id} 张图像，{ann_id} 个标注。输出文件：{output_json}")