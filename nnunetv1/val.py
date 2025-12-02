# validate_with_existing_weights.py
import torch
import os
import numpy as np
from collections import OrderedDict
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn

class WeightValidator(nnUNetTrainerV2):
    """
    精简的权重验证器，只包含验证功能，不包含训练
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None):
        # 调用父类初始化，但跳过训练相关的设置
        super().__init__(plans_file, fold, output_folder, dataset_directory)
        
        # 禁用训练相关功能
        self.training = False
        
    def initialize_for_validation(self):
        """只初始化验证所需的部分"""
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)
            
            # 加载plans文件
            self.load_plans_file()
            self.process_plans(self.plans)
            
            # 设置数据增强参数（用于验证时的预处理）
            self.setup_DA_params()
            
            # 初始化网络
            self.initialize_network()
            
            # 获取数据加载器
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] + "_stage%d" % self.stage)
            self.dl_tr, self.dl_val = self.get_basic_generators()
            
            self.was_initialized = True
            self.print_to_log_file("Validator initialized successfully")
    
    def load_weights(self, weights_path):
        """加载权重文件"""
        if not isfile(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.print_to_log_file(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 处理可能的DataParallel包装
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove 'module.' prefix
                else:
                    name = k
                new_state_dict[name] = v
            
            # 加载权重
            self.network.load_state_dict(new_state_dict)
            self.print_to_log_file("✅ Weights loaded successfully")
            
            # 打印权重信息
            if 'epochs_used' in checkpoint:
                self.print_to_log_file(f"Epochs used for averaging: {checkpoint['epochs_used']}")
            if 'val_losses' in checkpoint:
                best_loss = min(checkpoint['val_losses'])
                self.print_to_log_file(f"Best validation loss: {best_loss:.4f}")
        else:
            # 如果没有state_dict，假设文件直接包含权重
            self.network.load_state_dict(checkpoint)
            self.print_to_log_file("✅ Weights loaded successfully (direct state_dict)")
    
    def validate_with_loaded_weights(self, validation_folder_name='validation_loaded_weights'):
        """使用加载的权重进行验证"""
        self.print_to_log_file(f"Starting validation with folder name: {validation_folder_name}")
        
        # 设置网络为评估模式
        self.network.eval()
        
        # 运行验证
        results = self.validate(
            do_mirroring=True,
            use_sliding_window=True,
            step_size=0.5,
            save_softmax=True,
            use_gaussian=True,
            overwrite=True,
            validation_folder_name=validation_folder_name,
            debug=False,
            all_in_gpu=False
        )
        
        return results

def validate_single_weights(weights_path, plans_file, fold=0, output_folder=None, dataset_directory=None):
    """
    验证单个权重文件
    
    Args:
        weights_path: 权重文件路径
        plans_file: plans文件路径
        fold: 交叉验证的fold
        output_folder: 输出文件夹
        dataset_directory: 数据集目录
    """
    
    print(f"🔍 Validating weights: {weights_path}")
    
    # 如果未指定输出文件夹，使用权重文件所在目录
    if output_folder is None:
        output_folder = os.path.dirname(weights_path)
    
    # 创建验证器实例
    validator = WeightValidator(
        plans_file=plans_file,
        fold=fold,
        output_folder=output_folder,
        dataset_directory=dataset_directory
    )
    
    try:
        # 初始化验证器
        validator.initialize_for_validation()
        
        # 加载权重
        validator.load_weights(weights_path)
        
        # 运行验证
        validation_folder_name = f"validation_{os.path.basename(weights_path).replace('.pth', '')}"
        results = validator.validate_with_loaded_weights(validation_folder_name)
        
        print(f"✅ Validation completed! Results saved in: {join(output_folder, validation_folder_name)}")
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_multiple_weights(weights_dir, plans_file, fold=0, output_folder=None, dataset_directory=None):
    """
    验证目录下的多个权重文件
    
    Args:
        weights_dir: 包含权重文件的目录
        plans_file: plans文件路径
        fold: 交叉验证的fold
        output_folder: 输出文件夹
        dataset_directory: 数据集目录
    """
    
    print(f"🔍 Validating multiple weights in: {weights_dir}")
    
    # 查找所有的.pth文件
    weight_files = []
    for file in os.listdir(weights_dir):
        if file.endswith('.pth'):
            weight_files.append(join(weights_dir, file))
    
    if not weight_files:
        print("❌ No weight files found!")
        return
    
    print(f"Found {len(weight_files)} weight files")
    
    results = {}
    for weight_file in weight_files:
        print(f"\n{'='*50}")
        print(f"Validating: {os.path.basename(weight_file)}")
        print(f"{'='*50}")
        
        result = validate_single_weights(
            weights_path=weight_file,
            plans_file=plans_file,
            fold=fold,
            output_folder=output_folder,
            dataset_directory=dataset_directory
        )
        
        results[weight_file] = result
    
    return results

def main():
    """主函数 - 配置并运行验证"""
    
    # ========== 配置区域 ==========
    # 权重文件路径（单个文件或目录）
    WEIGHTS_PATH = "/data/dusy/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/averaged_model_final.pth"
    
    # Plans文件路径
    PLANS_FILE = "/data/dusy/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl"
    
    # 数据集目录
    DATASET_DIRECTORY = "/data/dusy/nnUNet/nnUNet_preprocessed/Task004_Hippocampus"
    
    # Fold编号
    FOLD = 0
    
    # 输出目录（可选，默认为权重文件所在目录）
    OUTPUT_FOLDER = None
    # =============================
    
    print("🚀 Starting Weight Validation")
    print("=" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Weights path does not exist: {WEIGHTS_PATH}")
        return
    
    if not os.path.exists(PLANS_FILE):
        print(f"❌ Plans file does not exist: {PLANS_FILE}")
        return
    
    # 判断是单个文件还是目录
    if os.path.isfile(WEIGHTS_PATH):
        # 验证单个权重文件
        validate_single_weights(
            weights_path=WEIGHTS_PATH,
            plans_file=PLANS_FILE,
            fold=FOLD,
            output_folder=OUTPUT_FOLDER,
            dataset_directory=DATASET_DIRECTORY
        )
    else:
        # 验证目录下的所有权重文件
        validate_multiple_weights(
            weights_dir=WEIGHTS_PATH,
            plans_file=PLANS_FILE,
            fold=FOLD,
            output_folder=OUTPUT_FOLDER,
            dataset_directory=DATASET_DIRECTORY
        )
    
    print("\n🎉 All validations completed!")

if __name__ == "__main__":
    main()