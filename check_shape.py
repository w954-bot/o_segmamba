import os
import SimpleITK as sitk

def check_shapes(root_folder):
    """
    检查指定文件夹下所有子文件夹中的 t1.nii.gz 和 seg.nii.gz 形状是否一致。
    (使用 SimpleITK 版本，不依赖 nibabel)
    """
    
    # 存储结果的列表
    mismatch_list = []  # 形状不一致的文件夹
    missing_list = []   # 文件缺失的文件夹
    error_list = []     # 文件损坏无法读取的文件夹
    consistent_count = 0 # 计数一致的数量

    # 获取主文件夹下所有的子内容
    if not os.path.exists(root_folder):
        print(f"错误: 文件夹 '{root_folder}' 不存在。")
        return

    subdirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    total_folders = len(subdirs)

    print(f"开始检查 '{root_folder}' 下的 {total_folders} 个子文件夹...\n")

    # 初始化 SimpleITK 的文件读取器
    reader = sitk.ImageFileReader()

    for subdir in subdirs:
        current_path = os.path.join(root_folder, subdir)
        
        t1_path = os.path.join(current_path, "t1.nii.gz")
        seg_path = os.path.join(current_path, "seg.nii.gz")

        # 1. 检查文件是否存在
        if not os.path.exists(t1_path) or not os.path.exists(seg_path):
            missing_files = []
            if not os.path.exists(t1_path): missing_files.append("t1.nii.gz")
            if not os.path.exists(seg_path): missing_files.append("seg.nii.gz")
            missing_list.append(f"{subdir} (缺失: {', '.join(missing_files)})")
            continue

        # 2. 检查形状是否一致
        try:
            # 使用 SimpleITK 读取图像信息 (ReadImageInformation 只读头文件，不读像素数据，速度快)
            
            # 读取 t1
            reader.SetFileName(t1_path)
            reader.ReadImageInformation()
            t1_shape = reader.GetSize() # 返回的是 (x, y, z) 元组

            # 读取 seg
            reader.SetFileName(seg_path)
            reader.ReadImageInformation()
            seg_shape = reader.GetSize()

            if t1_shape != seg_shape:
                mismatch_list.append({
                    "folder": subdir,
                    "t1_shape": t1_shape,
                    "seg_shape": seg_shape
                })
            else:
                consistent_count += 1

        except Exception as e:
            error_list.append(f"{subdir} (读取错误: {str(e)})")

    # --- 打印报告 ---
    print("=" * 50)
    print("检查报告 summary")
    print("=" * 50)

    # 1. 优先打印不一致的情况
    if len(mismatch_list) > 0:
        print(f"❌ 发现 {len(mismatch_list)} 个文件夹形状不一致：")
        for item in mismatch_list:
            print(f"   - 文件夹: {item['folder']}")
            print(f"     t1 : {item['t1_shape']}")
            print(f"     seg: {item['seg_shape']}")
    
    # 2. 打印文件缺失的情况
    if len(missing_list) > 0:
        print(f"\n⚠️  发现 {len(missing_list)} 个文件夹文件缺失：")
        for item in missing_list:
            print(f"   - {item}")

    # 3. 打印文件损坏的情况
    if len(error_list) > 0:
        print(f"\n🚫 发现 {len(error_list)} 个文件夹文件损坏：")
        for item in error_list:
            print(f"   - {item}")

    # 4. 最终判定
    print("-" * 50)
    if len(mismatch_list) == 0 and len(missing_list) == 0 and len(error_list) == 0:
        print(f"✅ 全部一致！所有 {total_folders} 个文件夹中的 t1 和 seg 形状均匹配。")
    else:
        print(f"检查完成。共检查 {total_folders} 个文件夹，其中 {consistent_count} 个完全一致。")
        print("请检查上方列出的问题文件夹。")

if __name__ == "__main__":
    # 在这里修改为你的实际数据文件夹路径
    # 例如: data_folder = "/home/user/data/dataset"
    data_folder = "./my_dataset_folder" 
    
    # 为了方便测试，如果你没有修改上面的路径，这里允许用户手动输入
    if not os.path.exists(data_folder):
        data_folder = input("请输入包含数据的文件夹路径: ").strip().strip('"').strip("'")

    check_shapes(data_folder)