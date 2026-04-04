import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class RFUAV_DatasetBuilder:
    def __init__(self, root_dir="e:/Myprojects/RF-Vision-UAV-Tracker/Drone RF Data", output_dir="e:/Myprojects/RF-Vision-UAV-Tracker/yolo_dataset"):
        self.root_dir = root_dir
        self.output_dir = output_dir
        
        # YOLO 标准组织形式
        self.img_train_dir = os.path.join(output_dir, "images/train")
        self.img_val_dir = os.path.join(output_dir, "images/val")
        self.lbl_train_dir = os.path.join(output_dir, "labels/train")
        self.lbl_val_dir = os.path.join(output_dir, "labels/val")
        
        for d in [self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lbl_val_dir]:
            os.makedirs(d, exist_ok=True)
            
        # 扫描出所有有效类别
        self.classes = []
        for name in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, name)):
                self.classes.append(name)
        print(f"🤖 侦测到 {len(self.classes)} 种无人机模型用于训练: {self.classes}")
        
        self.n_fft = 1024
        self.target_w = 640
        self.target_h = 640
        
        # 参数公式：100MSPS 下，1s 包含 100M 复数点。
        # 连续作 N_fft = 1024 的 FFT 会产生 100M/1024 = 97656 行频谱。
        # 降维至 YOLO 视界的 640 行（每组池化合并比率）：97656 // 640 = 152
        self.time_pool_size = 152
        # 在频宽两侧斩去多余频点只看中央 (1024 - 640) / 2 = 192
        self.freq_trim = (self.n_fft - self.target_w) // 2

    def process_iq_file(self, file_path, output_img_path):
        """
        核心物理：通过分块免爆内存读取 32 位交织复数并通过 MaxPool 防止突发信号截断。
        """
        data = np.fromfile(file_path, dtype=np.float32)
        complex_data = data[0::2] + 1j * data[1::2]
        total_samples = len(complex_data)
        
        valid_samples = (total_samples // self.n_fft) * self.n_fft
        complex_data = complex_data[:valid_samples]
        
        num_rows = valid_samples // self.n_fft
        stft_matrix = complex_data.reshape((num_rows, self.n_fft))
        
        window = np.blackman(self.n_fft)
        stft_matrix = stft_matrix * window
        fft_batch = np.fft.fftshift(np.fft.fft(stft_matrix, axis=1), axes=1)
        
        power_db = 20 * np.log10(np.abs(fft_batch) + 1e-12)
        power_db = power_db[:, self.freq_trim : -self.freq_trim]
        
        num_pooled_rows = num_rows // self.time_pool_size
        power_db = power_db[:num_pooled_rows * self.time_pool_size, :]
        reshaped_for_pooling = power_db.reshape((num_pooled_rows, self.time_pool_size, self.target_w))
        waterfall = np.max(reshaped_for_pooling, axis=1)
        
        waterfall_resized = cv2.resize(waterfall, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        
        plt.imsave(output_img_path, waterfall_resized, cmap='hot', vmin=30, vmax=110)
        
    def build(self):
        import random
        
        counter = 0
        for class_id, class_name in enumerate(self.classes):
            iq_files = glob.glob(os.path.join(self.root_dir, class_name, "**", "*.iq"), recursive=True)
            print(f"📦 正在剥离处理 [{class_name}] 数据，共 {len(iq_files)} 个原始切片...")
            
            for iq_file in iq_files:
                is_train = random.random() < 0.8
                split_img_dir = self.img_train_dir if is_train else self.img_val_dir
                split_lbl_dir = self.lbl_train_dir if is_train else self.lbl_val_dir
                
                base_name = f"uav_{class_id}_{os.path.basename(iq_file).split('.')[0]}_{counter}"
                counter += 1
                
                img_path = os.path.join(split_img_dir, base_name + ".jpg")
                txt_path = os.path.join(split_lbl_dir, base_name + ".txt")
                
                self.process_iq_file(iq_file, img_path)
                
                with open(txt_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                    
        print(f"\n🎉 完美收工！已处理转换 {counter} 帧有效数据集。")

        yaml_path = os.path.join(self.output_dir, "rf_uav.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"train: {self.img_train_dir.replace('\\', '/')}\n")
            f.write(f"val: {self.img_val_dir.replace('\\', '/')}\n\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")
        print(f"📝 YAML 已生成 -> {yaml_path}")
        
if __name__ == "__main__":
    builder = RFUAV_DatasetBuilder()
    builder.build()
