#Evaluation code

import os
import pandas as pd
import motmetrics as mm

gt_root = r'/data2/cjj/Re-SDM-Car-U-3/test_data'
pred_dir = r'/data2/cjj/trytracker/MP2Net/512unstab3/DLADCN/results/trackingResultsDLADCN' 

# ========================
def read_gt(file_path):
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'mark', 'label', 'vis', 'conf']
    df = pd.read_csv(file_path, header=None, names=columns)
   # df = df[df['mark'] == 1]
    return df[['frame', 'id', 'x', 'y', 'w', 'h']]

def read_pred(file_path):
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'label', 'vis']
    df = pd.read_csv(file_path, header=None, names=columns)
    return df[['frame', 'id', 'x', 'y', 'w', 'h']]

def to_mot_format(df):
    df = df.copy()
    df['x1'] = df['x']
    df['y1'] = df['y']
    df['x2'] = df['x'] + df['w']
    df['y2'] = df['y'] + df['h']
    return df[['frame', 'id', 'x1', 'y1', 'x2', 'y2']]

def evaluate_sequence(gt_df, pred_df):
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(gt_df['frame'].unique())
    for frame in frames:
        gt_f = gt_df[gt_df['frame'] == frame]
        pred_f = pred_df[pred_df['frame'] == frame]

        gt_ids = gt_f['id'].tolist()
        pred_ids = pred_f['id'].tolist()
        gt_boxes = gt_f[['x1', 'y1', 'x2', 'y2']].values
        pred_boxes = pred_f[['x1', 'y1', 'x2', 'y2']].values
        # if len(gt_ids) > 0 or len(pred_ids) > 0:
        #     print(f"Frame {frame}: GT {len(gt_ids)} boxes, Pred {len(pred_ids)} boxes")
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dist)

    return acc

def run_all_evaluations(gt_root, pred_dir):
    mh = mm.metrics.create()
    all_accs = {}

    for seq_name in os.listdir(gt_root):
        gt_seq_dir = os.path.join(gt_root, seq_name)
        gt_path = os.path.join(gt_seq_dir, 'gt.txt')
        print(gt_path)
        pred_path = os.path.join(pred_dir, f"{seq_name}.txt")
        print(pred_path)
                                                                            
        if not os.path.isdir(gt_seq_dir) or not os.path.exists(gt_path):
            continue
        if not os.path.exists(pred_path):
            print(f"缺少预测文件：{seq_name}.txt")
            continue

        print(f"正在评估：{seq_name}")
        try:
            gt_df = to_mot_format(read_gt(gt_path))
            pred_df = to_mot_format(read_pred(pred_path))
            acc = evaluate_sequence(gt_df, pred_df)
            all_accs[seq_name] = acc
        except Exception as e:
            print(f"评估失败：{seq_name}，错误：{e}")

    if not all_accs:
        print("没有成功评估任何序列。")
        return

    # 汇总评估
    summary = mh.compute_many(
        list(all_accs.values()),
        names=list(all_accs.keys()),
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )

    summary.to_csv("summary_512_U3_results.csv")
    pd.set_option('display.width', 1000)  
    pd.set_option('display.max_columns', None)  

    print("\n📊 所有评估完成，结果保存在 summary_results.csv")
    print(summary)

# ========================
if __name__ == '__main__':
    run_all_evaluations(gt_root, pred_dir)
