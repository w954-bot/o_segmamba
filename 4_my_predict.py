import os
import numpy as np
import torch

from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.prediction import Predictor
import nibabel as nib

set_determinism(123)

data_dir = "./data/fullres/train"
env = "pytorch"
max_epoch = 600
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [64, 224, 224]


class SingleModalBinaryTrainer(Trainer):
    """
    单模态(1通道) + 单目标(二分类) 分割测试/验证用 Trainer
    """

    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)

        self.patch_size = patch_size
        self.augmentation = False

        # ✅ 只加载一次模型/预测器
        self.model, self.predictor, self.save_path = self.define_model()
        # self.window_infer = SlidingWindowInferer(
        #     roi_size=self.patch_size,
        #     sw_batch_size=2,
        #     overlap=0.5,
        #     progress=True,
        #     mode="gaussian"
        # )


    def get_input(self, batch):
        """
        假设 dataloader 输出:
          batch["data"]: [B, 1, D, H, W]  (单模态)
          batch["seg"] : [B, 1, D, H, W]  (0/1)
        """
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]

        # 统一成 0/1 float
        label = (label > 0).float()
        return image, label, properties

    def define_model(self):
        """
        这里仍然用 SegMamba 举例：
        - in_chans=1  (单模态)
        - out_chans=2 (背景/前景)
        """
        from model_segmamba.segmamba import SegMamba

        model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384]
        )

        # TODO: 换成你自己的权重路径
        model_path = "logs/segmamba_windows/model/best_model_0.7819.pt"

        sd = torch.load(model_path, map_location="cpu")
        sd = self.filter_state_dict(sd)
        model.load_state_dict(sd, strict=True)

        model.to(self.device)
        model.eval()

        self.window_infer = SlidingWindowInferer(
            roi_size=self.patch_size,
            sw_batch_size=2,
            overlap=0.5,
            progress=True,
            mode="gaussian"
        )

        predictor = Predictor(
            window_infer=self.window_infer,
            mirror_axes=[0, 1, 2]  # 不想TTA可改成 None 或 []
        )

        save_path = "./prediction_results/best_7819"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    @torch.no_grad()
    # def validation_step(self, batch):
    #     image, label, properties = self.get_input(batch)

    #     # Predictor 内部做 sliding window + (可选) mirror TTA
    #     logits_or_prob = self.predictor.maybe_mirror_and_predict(image, self.model, device=self.device)

    #     # 通常这里会把网络输出整理到 “raw probability” 的格式（依你的 predictor 实现）
    #     prob = self.predictor.predict_raw_probability(logits_or_prob, properties=properties)
    #     # 期望: prob shape 为 [2, D, H, W] (C, D, H, W)

    #     # 二分类预测
    #     pred_cls = prob.argmax(dim=0)          # [D, H, W] 取 0/1
    #     pred_fg = (pred_cls == 1).float()      # 前景mask

    #     # 计算 Dice(前景)
    #     gt_fg = label[0, 0]  # [D,H,W]

    #     print("IMAGE:", image.shape)
    #     print("LABEL:", label.shape)
    #     print("RAW prob:", prob.shape)

    #     prob_nc = self.predictor.predict_noncrop_probability(prob, properties)
    #     print("NONCROP prob:", prob_nc.shape)


    #     d = dice(pred_fg.cpu().numpy(), gt_fg.cpu().numpy())
    #     print(f"[{properties['name'][0]}] Dice(FG) = {d:.4f}")

    #     # 保存结果（保存为 mask：0/1）
    #     # 如果你的 save_to_nii 需要原图尺寸映射，就走 noncrop；否则可直接保存 pred_fg
    #     pred_to_save = pred_fg[None]  # [1, D, H, W]
    #     pred_to_save = self.predictor.predict_noncrop_probability(pred_to_save, properties)

    #     # raw_spacing 建议从 properties 里拿；这里保底
    #     raw_spacing = properties.get("spacing", [1, 1, 1])
    #     if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) > 0 and isinstance(raw_spacing[0], (list, tuple)):
    #         # 有些实现 spacing 会是 batch list
    #         raw_spacing = raw_spacing[0]

    #     self.predictor.save_to_nii(
    #         pred_to_save.cpu(),
    #         raw_spacing=raw_spacing,
    #         case_name=properties["name"][0],
    #         save_dir=self.save_path
    #     )

    #     return d

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)

        image = image.to(self.device)
        label = (label > 0).float()  # [B,1,D,H,W]

        # ✅ 滑窗推理：输出空间应该和 image 一致
        logits = self.window_infer(image, self.model)  # 期望 [B,2,D,H,W] (二分类)

        # 兼容一下维度（有的模型可能返回 [B,C, ...]；正常就是5维）
        if logits.dim() != 5:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        prob = torch.softmax(logits, dim=1)            # [B,2,D,H,W]
        pred = prob.argmax(dim=1).float()              # [B,D,H,W] 0/1
        gt   = label[:, 0]                             # [B,D,H,W]

        # ✅ 此时 pred 和 gt 的 D/H/W 应该一致
        print("PRED:", pred[0].shape, "GT:", gt[0].shape)

        d = dice(pred[0].cpu().numpy(), gt[0].cpu().numpy())
        print(f"[{properties['name'][0]}] Dice = {d:.4f}")

        # 简单保存 nii（如果你数据本身没有 affine/spacing，就先用单位矩阵）
        save_dir = self.save_path
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, properties["name"][0] + ".nii.gz")

        pred_np = pred[0].cpu().numpy().astype(np.uint8)   # (D,H,W)
        nib.save(nib.Nifti1Image(pred_np, np.eye(4)), out_path)

        return d


    def filter_state_dict(self, sd):
        # 兼容 sd 是 {"module": ...} 或者 DDP 前缀 "module."
        if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module.") else k
            new_sd[new_k] = v
        return new_sd


if __name__ == "__main__":
    trainer = SingleModalBinaryTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir="",
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17751,
        training_script=__file__,
    )

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    # 测试集逐case推理/验证
    trainer.validation_single_gpu(test_ds)
