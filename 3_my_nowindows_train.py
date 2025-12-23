import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
set_determinism(123)
import os

# data_dir = "./data/fullres/try"
# logdir = f"./logs/segmamba_try"
data_dir = "./data/fullres/train"
logdir = f"./logs/segmamba"

model_save_path = os.path.join(logdir, "model")
# augmentation = "nomirror"
augmentation = True

env = "pytorch"
max_epoch = 4
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
roi_size = [64, 224, 224]

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation
        from model_segmamba.segmamba import SegMamba

        # Single-modality input (1 channel) and two-class output (background + tumor)
        self.model = SegMamba(in_chans=1,
                        out_chans=2,
                        depths=[2,2,2,2],
                        feat_size=[48, 96, 192, 384])

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.train_process = 18
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)

        # Softmax Dice for 2-class segmentation (background, tumor)
        self.loss_fn = DiceLoss(sigmoid=False, softmax=True, to_onehot_y=True, squared_pred=True, reduction="mean")
        self.scheduler_type = "poly"

    def training_step(self, batch):
        image, label = self.get_input(batch)
        
        pred = self.model(image)

        if len(label.shape) == 4:
            label = label.unsqueeze(1)

        loss = self.loss_fn(pred, label)

        self.log("training_loss", loss, step=self.global_step)

        return loss 
    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
    
        # Expect shape [B, 1, ...]; convert to class indices for softmax Dice
        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.window_infer(image, self.model)
        output = self.model(image)

        output = torch.softmax(output, dim=1).argmax(dim=1)

        output = output.cpu().numpy().astype(np.uint8)
        target = label.cpu().numpy().astype(np.uint8)
        
        pred_fg = output == 1
        target_fg = target == 1

        cal_dice, _ = self.cal_metric(target_fg, pred_fg)
        return cal_dice
    
    def validation_end(self, val_outputs):
        dice_score = val_outputs.mean()

        self.log("dice", dice_score, step=self.epoch)

        if dice_score > self.best_mean_dice:
            self.best_mean_dice = dice_score
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{dice_score:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{dice_score:.4f}.pt"), 
                                        delete_symbol="final_model")


        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{dice_score:.4f}.pt"))

        print(f"dice_score is {dice_score}")

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17759,
                            training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
