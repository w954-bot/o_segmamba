"查看训练集、验证集和测试集包含的文件列表"
import pickle
import os
import numpy as np
from tqdm import tqdm 
from torch.utils.data import Dataset 
import glob 
from light_training.dataloading.utils import unpack_dataset
import random 
from monai.utils import set_determinism
set_determinism(123)

class MedicalDataset(Dataset):
    def __init__(self, datalist, test=False) -> None:
        super().__init__()
        
        self.datalist = datalist
        self.test = test 

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(p)

            self.data_cached.append(info)

        ## unpacking
        print(f"unpacking data ....")
        # for 
        folder = []
        for p in self.datalist:
            f = os.path.dirname(p)
            if f not in folder:
                folder.append(f)
        for f in folder:
            unpack_dataset(f, 
                        unpack_segmentation=True,
                        overwrite_existing=False,
                        num_processes=8)


        print(f"data length is {len(self.datalist)}")
        
    def load_pkl(self, data_path):
        pass 
        properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)

        return info 
    
    def post(self, batch_data):
        return batch_data
    
    def read_data(self, data_path):
        
        image_path = data_path.replace(".npz", ".npy")
        seg_path = data_path.replace(".npz", "_seg.npy")
        image_data = np.load(image_path, "r+")
      
        seg_data = None 
        if not self.test:
            seg_data = np.load(seg_path, "r+")
        return image_data, seg_data

    def __getitem__(self, i):
        
        image, seg = self.read_data(self.datalist[i])

        properties = self.data_cached[i]

        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)
    

def get_train_val_test_loader_from_train(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    # random_state = random.random
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:] 

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader
if __name__ == '__main__':
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train("data/fullres/train")
    train_files = train_ds.datalist
    val_files = val_ds.datalist
    test_files = test_ds.datalist

    output_txt_name = "dataset_split_info.txt"
    
    with open(output_txt_name, "w", encoding='utf-8') as f:
        # 写入训练集
        f.write(f"=== Training Set (Count: {len(train_files)}) ===\n")
        for file_path in train_files:
            f.write(f"{file_path}\n")
        f.write("\n") # 空行分隔

        # 写入验证集
        f.write(f"=== Validation Set (Count: {len(val_files)}) ===\n")
        for file_path in val_files:
            f.write(f"{file_path}\n")
        f.write("\n")

        # 写入测试集
        f.write(f"=== Test Set (Count: {len(test_files)}) ===\n")
        for file_path in test_files:
            f.write(f"{file_path}\n")

    print(f"成功！所有文件名已保存到当前目录下的 {output_txt_name}")