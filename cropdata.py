# 批量化处理
# 批量处理多个视频的裁剪脚本 - 支持浮点数移动 + 标注过滤

import os
import cv2
import numpy as np
import random
import glob
from pathlib import Path


# ===== 固定配置参数 =====
BASE_INPUT_DIR = r'F:\SDM-Car\test_jpgs'
BASE_OUTPUT_DIR = r'F:\SDM-Car-U-2\test_jpgs'
CROP_SIZE = (512, 512)
IMG_SIZE = (1920, 1080)

# ===== 预设 16 个（起始点，方向）组合 - 支持浮点数移动 =====
# #2
START_CONFIGS = [
    ((100, 100), (2, 0.0)),  # 向右
    ((900, 300), (2, 0.0)),  # 向右
    ((500, 0), (0.0, 2)),  # 向下
    ((800, 100), (0.0, 2)),  # 向下
    ((200, 150), (1.42, 1.42)),  # 右下
    ((900, 50), (1.42, 1.42)),  # 右下
    ((900, 100), (-2, 0.0)),  # 向左
    ((1200, 400), (-2, 0.0)),  # 向左
    ((500, 450), (0.0, -2)),  # 向上
    ((1000, 450), (0.0, -2)),  # 向上
    ((1000, 500), (-1.42, -1.42)),  # 左上
    ((500, 400), (-1.42, -1.42)),  # 左上
    ((500, 500), (1.42, -1.42)),  # 右上
    ((900, 400), (1.42, -1.42)),  # 右上
    ((500, 50), (-1.42, 1.42)),  # 左下
    ((1000, 150), (-1.42, 1.42)),  # 左下
]


def find_video_folders():
    """扫描基础目录，找到所有视频文件夹"""
    video_folders = []
    for item in os.listdir(BASE_INPUT_DIR):
        item_path = os.path.join(BASE_INPUT_DIR, item)
        if os.path.isdir(item_path):
            # 检查是否包含img1文件夹和标注文件
            img1_path = os.path.join(item_path, 'img1')
            if os.path.exists(img1_path):
                # 查找标注文件（假设格式为 *-gt-*.txt）
                gt_files = glob.glob(os.path.join(item_path, '*-gt-*.txt'))
                if gt_files:
                    video_folders.append({
                        'name': item,
                        'path': item_path,
                        'img_dir': img1_path,
                        'label_path': gt_files[0]  # 取第一个找到的标注文件
                    })
                    print(f"✅ 找到视频文件夹: {item}")
                else:
                    print(f"⚠️  {item} 文件夹缺少标注文件，跳过")
            else:
                print(f"⚠️  {item} 文件夹缺少img1子文件夹，跳过")

    return video_folders


def read_annotations(label_path):
    """读取标注数据"""
    annotations = {}
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split(',')))
                frame_id = parts[0]
                annotations.setdefault(frame_id, []).append(parts)
        return annotations
    except Exception as e:
        print(f"❌ 读取标注文件失败: {label_path}, 错误: {e}")
        return {}


def apply_float_crop_warpaffine(img, crop_x, crop_y, crop_size):
    """使用 warpAffine 实现浮点数精度的裁剪"""
    # 计算整数部分和小数部分
    int_x = int(crop_x)
    int_y = int(crop_y)
    float_offset_x = crop_x - int_x
    float_offset_y = crop_y - int_y

    # 扩展裁剪区域以容纳浮点偏移
    extended_crop_size = (crop_size[0] + 1, crop_size[1] + 1)

    # 检查扩展区域是否超出图像边界
    if (int_x < 0 or int_y < 0 or
            int_x + extended_crop_size[0] >= img.shape[1] or
            int_y + extended_crop_size[1] >= img.shape[0]):
        return None, (crop_x, crop_y)

    # 先进行整数裁剪
    roi = img[int_y:int_y + extended_crop_size[1], int_x:int_x + extended_crop_size[0]]

    # 创建浮点平移变换矩阵
    translation_matrix = np.float32([
        [1, 0, -float_offset_x],
        [0, 1, -float_offset_y]
    ])

    # 应用 warpAffine 进行亚像素精度平移
    warped_roi = cv2.warpAffine(
        roi,
        translation_matrix,
        crop_size,
        flags=cv2.INTER_LINEAR,  # 双线性插值获得更好效果
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return warped_roi, (crop_x, crop_y)


def check_config_validity(annotations, frame_ids, start_point, move_direction):
    """检查配置是否在所有帧都有标注且不超出边界 - 预检查"""
    start_x, start_y = start_point
    dx, dy = move_direction

    for i, frame_id in enumerate(frame_ids):
        # 计算当前帧的浮点坐标
        current_x = start_x + dx * i
        current_y = start_y + dy * i

        # 🔍 检查边界 - 与 apply_float_crop_warpaffine 保持一致
        int_x = int(current_x)
        int_y = int(current_y)
        extended_crop_size = (CROP_SIZE[0] + 1, CROP_SIZE[1] + 1)

        if (int_x < 0 or int_y < 0 or
                int_x + extended_crop_size[0] >= IMG_SIZE[0] or
                int_y + extended_crop_size[1] >= IMG_SIZE[1]):
            return False, f"{i + 1} (边界超出: x={current_x:.1f}, y={current_y:.1f})"

        crop_x1, crop_y1 = current_x, current_y
        crop_x2, crop_y2 = current_x + CROP_SIZE[0], current_y + CROP_SIZE[1]

        # 检查这一帧是否有标注在裁剪区域内
        anns = annotations.get(frame_id, [])
        has_valid_annotation = False

        for ann in anns:
            _, track_id, x, y, w, h, *_ = ann
            box_x1, box_y1 = float(x), float(y)
            box_x2, box_y2 = float(x + w), float(y + h)

            # 计算交集
            inter_x1 = max(box_x1, crop_x1)
            inter_y1 = max(box_y1, crop_y1)
            inter_x2 = min(box_x2, crop_x2)
            inter_y2 = min(box_y2, crop_y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                # 转换到裁剪坐标系
                new_w = inter_x2 - inter_x1
                new_h = inter_y2 - inter_y1

                # 检查标注框是否足够大
                if int(round(new_w)) > 0 and int(round(new_h)) > 0:
                    has_valid_annotation = True
                    break

        if not has_valid_annotation:
            return False, f"{i + 1} (无标注)"

    return True, None


def process_single_video(video_info, selected_configs):
    """处理单个视频"""
    video_name = video_info['name']
    img_dir = video_info['img_dir']
    label_path = video_info['label_path']

    print(f"\n🎬 开始处理视频: {video_name}")

    # 读取标注数据
    annotations = read_annotations(label_path)
    if not annotations:
        print(f"❌ {video_name} 标注数据为空，跳过")
        return

    frame_ids = sorted(annotations.keys())
    num_frames = len(frame_ids)
    print(f"📊 视频 {video_name} 共有 {num_frames} 帧")

    # 为每个配置处理
    valid_configs = 0
    for idx, (start_point, move_direction) in enumerate(selected_configs):
        start_x, start_y = start_point
        dx, dy = move_direction
        print(f"🚀 处理配置 {idx + 1}: 起点=({start_x}, {start_y}) 方向=({dx}, {dy})")

        # 🔍 预检查：验证所有帧是否都有标注且不超出边界
        is_valid, failed_info = check_config_validity(annotations, frame_ids, start_point, move_direction)
        if not is_valid:
            print(f"❌ 配置 {idx + 1} 在第 {failed_info} 帧失败，跳过此配置")
            continue

        print(f"✅ 配置 {idx + 1} 预检查通过，所有帧都有标注且不超出边界")

        # 输出路径
        output_video_dir = os.path.join(BASE_OUTPUT_DIR, video_name)
        output_img_dir = os.path.join(output_video_dir, f'crop_{idx + 1}', 'img1')
        output_label_path = os.path.join(output_video_dir, f'crop_{idx + 1}', f'{video_name}-gt-crop.txt')

        os.makedirs(output_img_dir, exist_ok=True)
        out_annos = []

        processed_frames = 0
        for i, frame_id in enumerate(frame_ids):
            # 计算当前帧的浮点坐标（保持完整浮点精度）
            current_x = start_x + dx * i
            current_y = start_y + dy * i

            img_name = f"{frame_id:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                print(f"图像 {img_path} 未找到，跳过")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像 {img_path}，跳过")
                continue

            # 使用 warpAffine 进行浮点精度裁剪
            cropped_img, actual_pos = apply_float_crop_warpaffine(img, current_x, current_y, CROP_SIZE)

            if cropped_img is None:
                print(f"第{i + 1}帧裁剪窗口超出图像边界 (x={current_x:.3f}, y={current_y:.3f})，停止当前配置")
                break

            # 保存裁剪后的图像
            out_img_name = f"{i + 1:06d}.jpg"
            cv2.imwrite(os.path.join(output_img_dir, out_img_name), cropped_img)

            # 映射标注 - 使用浮点坐标进行精确计算
            anns = annotations.get(frame_id, [])
            for ann in anns:
                _, track_id, x, y, w, h, *_ = ann
                box_x1, box_y1 = float(x), float(y)
                box_x2, box_y2 = float(x + w), float(y + h)
                crop_x1, crop_y1 = current_x, current_y
                crop_x2, crop_y2 = current_x + CROP_SIZE[0], current_y + CROP_SIZE[1]

                # 计算交集
                inter_x1 = max(box_x1, crop_x1)
                inter_y1 = max(box_y1, crop_y1)
                inter_x2 = min(box_x2, crop_x2)
                inter_y2 = min(box_y2, crop_y2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # 转换到裁剪坐标系（保持浮点精度）
                    new_x = inter_x1 - crop_x1
                    new_y = inter_y1 - crop_y1
                    new_w = inter_x2 - inter_x1
                    new_h = inter_y2 - inter_y1

                    # 四舍五入到整数（仅用于标注保存）
                    new_x_int = int(round(new_x))
                    new_y_int = int(round(new_y))
                    new_w_int = int(round(new_w))
                    new_h_int = int(round(new_h))

                    # 确保标注框有效
                    if new_w_int > 0 and new_h_int > 0:
                        out_annos.append(
                            f"{i + 1},{track_id},{new_x_int},{new_y_int},{new_w_int},{new_h_int},-1,-1,-1,-1")

            processed_frames += 1

        # 保存标注文件
        with open(output_label_path, 'w') as f:
            for line in out_annos:
                f.write(line + '\n')

        # 保存配置信息
        config_info_path = os.path.join(os.path.dirname(output_label_path), 'crop_config.txt')
        with open(config_info_path, 'w') as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Start Point: ({start_x}, {start_y})\n")
            f.write(f"Move Direction: ({dx}, {dy})\n")
            f.write(f"Crop Method: warpAffine (Float Precision)\n")
            f.write(f"Processed Frames: {processed_frames}\n")
            f.write(f"Total Annotations: {len(out_annos)}\n")
            f.write(f"Annotation Filter: All frames have valid annotations\n")
            f.write(f"Boundary Check: All frames within image bounds\n")
            f.write(f"Float Position Info:\n")
            f.write(f"  Final float position: ({current_x:.3f}, {current_y:.3f})\n")
            f.write(f"  Interpolation: cv2.INTER_LINEAR\n")

        print(f"✅ 配置 {idx + 1} 处理完成！处理了 {processed_frames} 帧，生成 {len(out_annos)} 个标注")
        print(f"📍 最终浮点位置: ({current_x:.3f}, {current_y:.3f}) (warpAffine精度)")
        valid_configs += 1

    if valid_configs > 0:
        print(f"🎉 视频 {video_name} 处理完成！成功配置: {valid_configs}/{len(selected_configs)}")
    else:
        print(f"⚠️ 视频 {video_name} 所有配置都被过滤，没有生成输出")


def create_custom_config():
    """创建自定义配置的交互式函数"""
    print("\n🎨 创建自定义配置:")
    configs = []

    try:
        num_configs = int(input("请输入要创建的配置数量 (默认4): ") or "4")

        for i in range(num_configs):
            print(f"\n配置 {i + 1}:")
            start_x = float(input(f"  起始X坐标 (0-{IMG_SIZE[0] - CROP_SIZE[0]}): "))
            start_y = float(input(f"  起始Y坐标 (0-{IMG_SIZE[1] - CROP_SIZE[1]}): "))
            dx = float(input("  X方向移动速度 (可为负数，支持小数如1.5): "))
            dy = float(input("  Y方向移动速度 (可为负数，支持小数如1.2): "))

            configs.append(((start_x, start_y), (dx, dy)))
            print(f"  ✅ 配置 {i + 1}: 起点=({start_x}, {start_y}) 方向=({dx}, {dy})")

    except ValueError as e:
        print(f"❌ 输入格式错误: {e}")
        return None
    except KeyboardInterrupt:
        print("\n❌ 用户取消输入")
        return None

    return configs


def main():
    """主函数"""
    print("🔍 扫描视频文件夹...")
    video_folders = find_video_folders()

    if not video_folders:
        print("❌ 未找到任何有效的视频文件夹！")
        return

    print(f"📁 找到 {len(video_folders)} 个视频文件夹:")
    for i, video in enumerate(video_folders, 1):
        print(f"  {i}. {video['name']}")

    # 询问用户选择模式
    print(f"\n🎯 请选择裁剪配置模式:")
    print("  1. 所有视频使用相同的随机配置（可重复）")
    print("  2. 每个视频使用不同的随机配置")
    print("  3. 所有视频使用前4个固定配置")
    print("  4. 自定义配置（支持浮点数移动）")
    print("  🔧 使用 warpAffine 浮点平移方案，保持亚像素精度")
    print("  🎯 标注过滤: 只输出每帧都有标注且不超出边界的裁剪结果")

    try:
        mode = input("请输入选择 (1/2/3/4，默认为1): ").strip()
        if mode not in ['1', '2', '3', '4', '']:
            mode = '1'
        if mode == '':
            mode = '1'
    except:
        mode = '1'

    # 如果选择自定义模式，获取自定义配置
    custom_configs = None
    if mode == '4':
        custom_configs = create_custom_config()
        if custom_configs is None:
            print("❌ 自定义配置失败，退出程序")
            return

    # 处理每个视频
    total_videos = len(video_folders)
    for i, video_info in enumerate(video_folders, 1):
        print(f"\n{'=' * 60}")
        print(f"📹 进度: {i}/{total_videos} - 处理视频: {video_info['name']}")
        print(f"{'=' * 60}")

        # 根据模式选择配置
        if mode == '1':
            # 所有视频使用相同随机配置
            random.seed(42)  # 固定种子
            selected_configs = random.sample(START_CONFIGS, 4)
        elif mode == '2':
            # 每个视频使用不同随机配置
            random.seed(42 + i)  # 不同的种子
            selected_configs = random.sample(START_CONFIGS, 4)
        elif mode == '3':
            # 使用前4个固定配置
            selected_configs = START_CONFIGS[:4]
        else:  # mode == '4'
            # 使用自定义配置
            selected_configs = custom_configs

        print(f"🎯 视频 {video_info['name']} 的配置:")
        for j, (start_point, move_direction) in enumerate(selected_configs, 1):
            print(f"  {j}. 起点=({start_point[0]}, {start_point[1]}) 方向=({move_direction[0]}, {move_direction[1]})")

        try:
            process_single_video(video_info, selected_configs)
        except Exception as e:
            print(f"❌ 处理视频 {video_info['name']} 时出错: {e}")
            continue

    print(f"\n🎊 批量处理完成！共处理了 {total_videos} 个视频")
    print(f"📂 输出目录: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
