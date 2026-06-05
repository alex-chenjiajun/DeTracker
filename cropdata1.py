# Batch Processing of Mixed-Motion Video Cropping with Subpixel-Accurate WarpAffine

import os
import cv2
import numpy as np
import random
import glob
from pathlib import Path

# ===== 固定配置参数 =====
BASE_INPUT_DIR = r'F:\SDM-Car\test_jpgs'
BASE_OUTPUT_DIR = r'F:\SDM-Car-U-3\test_jpgs_mix'
CROP_SIZE = (512, 512)
IMG_SIZE = (1920, 1080)

# ===== 运动配置 =====s
MOTION_COMBINATIONS = [
    # 组合1：向右下区域运动
    [(3, 0), (0, 3), (1.732, 1.732)],
    # 组合2：向左上区域运动
    [(-3, 0), (0, -3), (-1.732, -1.732)],
    # 组合3：向右上区域运动
    [(3, 0), (0, -3), (1.732, -1.732)],
    # 组合4：向左下区域运动
    [(-3, 0), (0, 3), (-1.732, 1.732)],
]


# 每种运动模式的起始位置（x, y）
MOTION_START_POSITIONS = [
    (100, 100),  # 组合1：从左上角开始向右下运动
    (1000, 500),  # 组合2：从右下角开始向左上运动
    (200, 500),  # 组合3：从左下角开始向右上运动
    (1000, 100),  # 组合4：从右上角开始向左下运动
]

# 每种运动持续的帧数范围
MOTION_DURATION_RANGE = (20, 30)


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
            int_x + extended_crop_size[0] > img.shape[1] or
            int_y + extended_crop_size[1] > img.shape[0]):
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


def generate_motion_sequence(total_frames, mode_idx=None):
    """生成运动序列，整个数据集使用一种运动组合，每帧从组合的3个方向中随机选择"""
    # 如果未指定模式，随机选择一种运动模式
    if mode_idx is None:
        mode_idx = random.randint(0, len(MOTION_COMBINATIONS) - 1)

    motion_dirs = MOTION_COMBINATIONS[mode_idx]
    start_pos = MOTION_START_POSITIONS[mode_idx]

    print(f"选择运动模式 {mode_idx + 1}，起始位置: {start_pos}，方向组合: {motion_dirs}")

    motion_sequence = []

    # 设置起始位置标记
    for frame_idx in range(total_frames):
        # 从当前模式的3个方向中随机选择一个
        motion_dir = random.choice(motion_dirs)
        # 只有第一帧需要设置起始位置
        motion_sequence.append((motion_dir, start_pos if frame_idx == 0 else None))

    return motion_sequence, mode_idx


def calculate_positions(motion_sequence, crop_size, img_size):
    """根据运动序列计算每帧的裁剪位置（保持浮点数精度）"""
    positions = []
    x, y = 0.0, 0.0  # 使用浮点数保持精度

    for i, motion_data in enumerate(motion_sequence):
        if isinstance(motion_data, tuple) and len(motion_data) == 2:
            (dx, dy), start_pos = motion_data
            # 如果有新的起始位置，使用它
            if start_pos is not None:
                x, y = float(start_pos[0]), float(start_pos[1])
        else:
            # 兼容旧格式
            dx, dy = motion_data

        # 计算新位置（保持浮点数）
        new_x = x + float(dx)
        new_y = y + float(dy)

        # 边界检查和处理（使用扩展边界以适应warpAffine）
        int_x = int(new_x)
        int_y = int(new_y)
        extended_crop_size = (crop_size[0] + 1, crop_size[1] + 1)

        if int_x < 0:
            new_x = 0.0
        elif int_x + extended_crop_size[0] > img_size[0]:
            new_x = float(img_size[0] - extended_crop_size[0])

        if int_y < 0:
            new_y = 0.0
        elif int_y + extended_crop_size[1] > img_size[1]:
            new_y = float(img_size[1] - extended_crop_size[1])

        # 保存浮点数位置
        positions.append((new_x, new_y))
        x, y = new_x, new_y

    return positions


def check_config_validity(annotations, frame_ids, positions):
    """检查配置是否在所有帧都有标注且不超出边界 - 预检查"""
    for i, frame_id in enumerate(frame_ids):
        if i >= len(positions):
            return False, i + 1

        # 获取浮点坐标
        crop_x, crop_y = positions[i]

        # 检查边界 - 与 apply_float_crop_warpaffine 保持一致
        int_x = int(crop_x)
        int_y = int(crop_y)
        extended_crop_size = (CROP_SIZE[0] + 1, CROP_SIZE[1] + 1)

        if (int_x < 0 or int_y < 0 or
                int_x + extended_crop_size[0] > IMG_SIZE[0] or
                int_y + extended_crop_size[1] > IMG_SIZE[1]):
            return False, f"{i + 1} (边界超出: x={crop_x:.3f}, y={crop_y:.3f})"

        crop_x1, crop_y1 = crop_x, crop_y
        crop_x2, crop_y2 = crop_x + CROP_SIZE[0], crop_y + CROP_SIZE[1]

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
                new_w = inter_x2 - inter_x1
                new_h = inter_y2 - inter_y1

                # 检查标注框是否足够大
                if int(round(new_w)) > 0 and int(round(new_h)) > 0:
                    has_valid_annotation = True
                    break

        if not has_valid_annotation:
            return False, f"{i + 1} (无标注)"

    return True, None


def process_single_video(video_info, num_configs=4):
    """处理单个视频，生成多个混合运动配置"""
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

    valid_configs = 0
    for config_idx in range(num_configs):
        print(f"\n🚀 处理配置 {config_idx + 1}/{num_configs}")

        # 为每个配置设置不同的随机种子，确保可重现
        random.seed(42 + hash(video_name) + config_idx)

        # 生成运动序列
        motion_sequence, selected_mode = generate_motion_sequence(num_frames)

        # 计算所有帧的位置（浮点数）
        positions = calculate_positions(motion_sequence, CROP_SIZE, IMG_SIZE)

        # 预检查：验证所有帧是否都有标注
        is_valid, failed_info = check_config_validity(annotations, frame_ids, positions)
        if not is_valid:
            print(f"❌ 配置 {config_idx + 1} 在第 {failed_info} 帧失败，跳过此配置")
            continue

        print(f"✅ 配置 {config_idx + 1} 预检查通过，所有帧都有标注且不超出边界")
        print(f"🎯 运动模式: {selected_mode + 1} ({'右下区域,左上区域,右上区域,左下区域'.split(',')[selected_mode]})")

        # 输出路径
        output_video_dir = os.path.join(BASE_OUTPUT_DIR, video_name)
        # output_img_dir = os.path.join(output_video_dir, f'mixed_motion_{config_idx + 1}', 'img1')
        # output_label_path = os.path.join(output_video_dir, f'mixed_motion_{config_idx + 1}',
        #                                  f'{video_name}-gt-crop.txt')

        output_img_dir = os.path.join(output_video_dir, f'crop_{config_idx + 1}', 'img1')
        output_label_path = os.path.join(output_video_dir, f'crop_{config_idx + 1}',
                                         f'{video_name}-gt-crop.txt')

        os.makedirs(output_img_dir, exist_ok=True)
        out_annos = []

        processed_frames = 0
        motion_stats = {}

        # 逐帧处理
        for i, frame_id in enumerate(frame_ids):
            if i >= len(positions):
                print(f"位置序列不足，在第{i}帧停止")
                break

            # 获取当前帧的浮点坐标（保持完整浮点精度）
            crop_x, crop_y = positions[i]

            # 加载图像
            img_name = f"{frame_id:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                print(f"图像 {img_path} 未找到，跳过")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"无法读取图像 {img_path}，跳过")
                continue

            # 使用 warpAffine 进行浮点精度裁剪
            cropped_img, actual_pos = apply_float_crop_warpaffine(img, crop_x, crop_y, CROP_SIZE)

            if cropped_img is None:
                print(f"第{i + 1}帧裁剪窗口超出图像边界 (x={crop_x:.3f}, y={crop_y:.3f})，停止当前配置")
                break

            # 保存图像
            out_img_name = f"{i + 1:06d}.jpg"
            cv2.imwrite(os.path.join(output_img_dir, out_img_name), cropped_img)

            # 处理当前帧的标注 - 使用浮点坐标进行精确计算
            anns = annotations.get(frame_id, [])
            for ann in anns:
                _, track_id, x, y, w, h, *_ = ann

                # 框的四个边（使用浮点数）
                box_x1, box_y1 = float(x), float(y)
                box_x2, box_y2 = float(x + w), float(y + h)

                # 当前裁剪区域（浮点数）
                crop_x1, crop_y1 = crop_x, crop_y
                crop_x2, crop_y2 = crop_x + CROP_SIZE[0], crop_y + CROP_SIZE[1]

                # 判断是否与窗口相交
                inter_x1 = max(box_x1, crop_x1)
                inter_y1 = max(box_y1, crop_y1)
                inter_x2 = min(box_x2, crop_x2)
                inter_y2 = min(box_y2, crop_y2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # 有交集，转换到裁剪坐标系（保持浮点精度）
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

            # 统计运动信息
            if isinstance(motion_sequence[i], tuple) and len(motion_sequence[i]) == 2:
                motion, _ = motion_sequence[i]
            else:
                motion = motion_sequence[i]
            motion_stats[motion] = motion_stats.get(motion, 0) + 1

            processed_frames += 1

        # 保存标注文件
        with open(output_label_path, 'w') as f:
            for line in out_annos:
                f.write(line + '\n')

        # 保存配置信息和统计
        config_info_path = os.path.join(os.path.dirname(output_label_path), 'crop_config.txt')
        final_x, final_y = positions[-1] if positions else (0, 0)
        with open(config_info_path, 'w') as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Config: {config_idx + 1}\n")
            f.write(f"Motion Mode: {selected_mode + 1}\n")
            f.write(f"Mode Description: {['右下区域', '左上区域', '右上区域', '左下区域'][selected_mode]}\n")
            f.write(f"Start Position: {MOTION_START_POSITIONS[selected_mode]}\n")
            f.write(f"Available Directions: {MOTION_COMBINATIONS[selected_mode]}\n")
            f.write(f"Crop Method: warpAffine (Sub-pixel Precision)\n")
            f.write(f"Interpolation: cv2.INTER_LINEAR\n")
            f.write(f"Processed Frames: {processed_frames}\n")
            f.write(f"Total Annotations: {len(out_annos)}\n")
            f.write(f"Annotation Filter: All frames have valid annotations\n")
            f.write(f"Boundary Check: All frames within image bounds\n")
            f.write(f"Final Float Position: ({final_x:.3f}, {final_y:.3f})\n")
            f.write(f"\nMotion Statistics:\n")

            motion_names = {
                (3, 0): "向右", (0, 3): "向下", (1.732, 1.732): "右下",
                (-3, 0): "向左", (0, -3): "向上", (-1.732, -1.732): "左上",
                (1.732, -1.732): "右上", (-1.732, 1.732): "左下"
            }

            for motion, count in motion_stats.items():
                direction_name = motion_names.get(motion, f"方向{motion}")
                percentage = count / len(motion_sequence) * 100
                f.write(f"  {direction_name}: {count} 帧 ({percentage:.1f}%)\n")

        print(f"✅ 配置 {config_idx + 1} 处理完成！处理了 {processed_frames} 帧，生成 {len(out_annos)} 个标注")
        print(f"📍 运动模式 {selected_mode + 1}: {['右下区域', '左上区域', '右上区域', '左下区域'][selected_mode]}")
        print(f"📐 最终浮点位置: ({final_x:.3f}, {final_y:.3f}) (warpAffine亚像素精度)")

        # 输出运动统计
        print("🔄 运动统计:")
        for motion, count in motion_stats.items():
            direction_name = motion_names.get(motion, f"方向{motion}")
            percentage = count / len(motion_sequence) * 100
            print(f"   {direction_name}: {count}帧 ({percentage:.1f}%)")

        valid_configs += 1

    if valid_configs > 0:
        print(f"\n🎉 视频 {video_name} 处理完成！成功配置: {valid_configs}/{num_configs}")
    else:
        print(f"\n⚠️ 视频 {video_name} 所有配置都被过滤，没有生成输出")


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

    # 询问用户配置数量
    print(f"\n🎯 混合运动裁剪配置:")
    print("  🎲 每个视频将生成多个混合运动配置")
    print("  🔄 每个配置随机选择一种运动模式(右下/左上/右上/左下)")
    print("  🎯 每帧从该模式的3个方向中随机选择")
    print("  🛡️ 标注过滤: 只输出每帧都有标注的配置")
    print("  🔬 使用 warpAffine 实现亚像素精度裁剪")
    print("  📐 支持小数坐标运动，无需四舍五入")

    try:
        num_configs = int(input("请输入每个视频生成的配置数量 (默认4): ") or "4")
        if num_configs <= 0:
            num_configs = 4
    except:
        num_configs = 4

    print(f"📊 将为每个视频生成 {num_configs} 个混合运动配置")

    # 处理每个视频
    total_videos = len(video_folders)
    for i, video_info in enumerate(video_folders, 1):
        print(f"\n{'=' * 60}")
        print(f"📹 进度: {i}/{total_videos} - 处理视频: {video_info['name']}")
        print(f"{'=' * 60}")

        try:
            process_single_video(video_info, num_configs)
        except Exception as e:
            print(f"❌ 处理视频 {video_info['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎊 批量处理完成！共处理了 {total_videos} 个视频")
    print(f"📂 输出目录: {BASE_OUTPUT_DIR}")
    print(f"🎲 每个视频生成了最多 {num_configs} 个混合运动配置")
    print(f"🔄 运动模式说明:")
    print(f"   模式1: 右下区域运动 (向右/向下/右下)")
    print(f"   模式2: 左上区域运动 (向左/向上/左上)")
    print(f"   模式3: 右上区域运动 (向右/向上/右上)")
    print(f"   模式4: 左下区域运动 (向左/向下/左下)")
    print(f"🔬 裁剪方法: cv2.warpAffine 亚像素精度平移")


if __name__ == "__main__":
    main()