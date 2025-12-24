from light_training.dataloading.dataset import get_train_val_test_loader_from_train,get_loader_from_txt
from monai.utils import set_determinism
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm
import torch

set_determinism(123)

parser = argparse.ArgumentParser()
parser.add_argument("--pred_name", required=True, type=str)
args = parser.parse_args()

results_root = "prediction_results"
pred_name = args.pred_name


def to_binary_mask(arr, thr=0.5):
    """
    把预测/标签统一转成二值mask (numpy bool)
    - 如果是概率图(float)，用 thr 阈值
    - 如果是label图(int)，用 >0
    """
    arr = np.asarray(arr)
    if arr.dtype.kind in ["f"]:  # float
        return (arr >= thr)
    else:  # int/bool
        return (arr > 0)


def cal_metric_binary(gt_mask, pred_mask, voxel_spacing_zyx):
    """
    gt_mask/pred_mask: numpy bool, shape (D,H,W)
    voxel_spacing_zyx: (sz, sy, sx) 对应数组轴顺序
    """
    gt_sum = int(gt_mask.sum())
    pred_sum = int(pred_mask.sum())

    # 你原代码：只要有一个为空就给 (0,50)。
    # 更常见的做法是：两者都空 => (1,0) 表示完美
    if gt_sum == 0 and pred_sum == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    if gt_sum == 0 or pred_sum == 0:
        return np.array([0.0, 50.0], dtype=np.float32)

    dice = metric.binary.dc(pred_mask, gt_mask)
    hd95 = metric.binary.hd95(pred_mask, gt_mask, voxelspacing=voxel_spacing_zyx)
    return np.array([dice, hd95], dtype=np.float32)


if __name__ == "__main__":
    data_dir = "./data/fullres/train"
    test_ds = get_loader_from_txt('dataset_split_info.txt')[2]  # (train_ds, val_ds, test_ds)
    print("num test cases =", len(test_ds))

    all_results = np.zeros((len(test_ds), 2), dtype=np.float32)  # (N, [dice, hd95])

    for ind, batch in enumerate(tqdm(test_ds, total=len(test_ds))):
        properties = batch.get("properties", {})
        case_name = properties.get("name", f"case_{ind:04d}")

        # ---- GT：直接从test_ds里取seg（更通用，不依赖raw_data_dir）----
        gt = batch["seg"]
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        # 常见形状：(1,D,H,W) 或 (D,H,W)
        gt = np.squeeze(gt)
        gt = gt.transpose(2,1,0) # 再次 squeeze 防止 (1,D,H,W) -> (D,H,W)
        gt_mask = to_binary_mask(gt)

        # ---- Pred：从nii.gz读取 ----
        pred_path = f"./{results_root}/{pred_name}/{case_name}.nii.gz"
        pred_itk = sitk.ReadImage(pred_path)
        pred = sitk.GetArrayFromImage(pred_itk)  # (D,H,W) in zyx order (because GetArrayFromImage)

        # 有些人保存成 (1,D,H,W) 或 (C,D,H,W)，这里做个鲁棒处理
        pred = np.squeeze(pred)
        pred_mask = to_binary_mask(pred, thr=0.5)

        assert gt_mask.shape == pred_mask.shape, \
            f"Shape mismatch! GT: {gt_mask.shape}, Pred: {pred_mask.shape}"

        # ---- spacing：SimpleITK 是 (x,y,z)，数组是 (z,y,x) -> 反过来给 medpy ----
        spacing_xyz = pred_itk.GetSpacing()  # (sx, sy, sz)
        voxel_spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

        m = cal_metric_binary(gt_mask, pred_mask, voxel_spacing_zyx)
        all_results[ind] = m

    os.makedirs(f"./{results_root}/result_metrics/", exist_ok=True)
    save_path = f"./{results_root}/result_metrics/{pred_name}.npy"
    np.save(save_path, all_results)

    result = np.load(save_path)
    print("result shape:", result.shape)          # (N,2)
    print("mean [dice, hd95]:", result.mean(axis=0))
    print("std  [dice, hd95]:", result.std(axis=0))
