import os
import torch
import glob
import re
from collections import OrderedDict

def average_checkpoints_with_proper_saving(checkpoints_dir, output_path, num_best=30):
    """
    修复版本：确保保存设置与原始checkpoint一致
    """
    
    print(f"Looking for checkpoints in: {checkpoints_dir}")
    
    # 查找所有checkpoint文件
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return False
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # 提取epoch和loss信息
    checkpoint_info = []
    pattern = r'checkpoint_epoch_(\d+)\.pth'
    
    for checkpoint_file in checkpoint_files:
        match = re.search(pattern, os.path.basename(checkpoint_file))
        if match:
            epoch = int(match.group(1))
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                if 'state_dict' not in checkpoint:
                    continue
                val_loss = checkpoint.get('val_loss', float('inf'))
                checkpoint_info.append((epoch, val_loss, checkpoint_file))
            except Exception as e:
                print(f"Error loading {checkpoint_file}: {e}")
    
    if not checkpoint_info:
        print("No valid checkpoints found!")
        return False
    
    # 按loss排序，选择最佳的几个
    checkpoint_info.sort(key=lambda x: x[1])
    best_checkpoints = checkpoint_info[:num_best]
    
    print(f"\nSelected {len(best_checkpoints)} best checkpoints for averaging:")
    for i, (epoch, loss, path) in enumerate(best_checkpoints):
        print(f"  {i+1}. Epoch {epoch}: {loss:.4f}")
    
    # 分析原始checkpoint的保存格式
    print("\nAnalyzing original checkpoint format...")
    first_checkpoint_path = best_checkpoints[0][2]
    first_checkpoint = torch.load(first_checkpoint_path, map_location='cpu', weights_only=False)
    
    # 检查数据类型
    print("Checking data types in original checkpoint:")
    for key, tensor in list(first_checkpoint['state_dict'].items())[:5]:  # 检查前5个
        print(f"  {key}: {tensor.dtype}, size: {tuple(tensor.shape)}")
    
    # 开始平均权重
    print("\nAveraging weights...")
    
    averaged_state_dict = OrderedDict()
    num_loaded = 0
    
    # 初始化平均字典
    for key, tensor in first_checkpoint['state_dict'].items():
        # 保持原始数据类型
        averaged_state_dict[key] = torch.zeros_like(tensor, dtype=tensor.dtype)
    
    for epoch, val_loss, checkpoint_path in best_checkpoints:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['state_dict']
            
            # 累加权重
            for key in averaged_state_dict:
                if key in state_dict:
                    # 确保数据类型一致
                    averaged_state_dict[key] += state_dict[key].to(averaged_state_dict[key].dtype)
            
            num_loaded += 1
            print(f"✓ Loaded epoch {epoch}")
            
        except Exception as e:
            print(f"✗ Error processing epoch {epoch}: {e}")
    
    if num_loaded == 0:
        print("No checkpoints were successfully loaded!")
        return False
    
    # 计算平均值
    print(f"\nCalculating average over {num_loaded} checkpoints...")
    for key in averaged_state_dict:
        averaged_state_dict[key] = averaged_state_dict[key] / num_loaded
    
    # 保存平均模型 - 使用与原始checkpoint相同的结构
    print("\nSaving averaged model with proper format...")
    
    # 创建一个与原始checkpoint结构完全相同的字典
    save_dict = {
        'state_dict': averaged_state_dict,
        'epochs_used': [epoch for epoch, _, _ in best_checkpoints],
        'val_losses': [loss for _, loss, _ in best_checkpoints],
        'num_checkpoints': num_loaded,
        'epoch': best_checkpoints[-1][0],  # 添加epoch信息
        'optimizer': first_checkpoint.get('optimizer', None),  # 保持相同的结构
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 方法1: 使用与原始相同的保存选项
    print("Saving with torch.save (default options)...")
    torch.save(save_dict, output_path)
    
    # 检查文件大小
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024*1024)
        original_size = os.path.getsize(first_checkpoint_path) / (1024*1024)
        
        print(f"\n📊 File size comparison:")
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Averaged: {file_size:.2f} MB")
        print(f"  Ratio: {file_size/original_size:.2f}")
        
        # 如果文件大小仍然不匹配，尝试其他保存方法
        if file_size < original_size * 0.8:  # 如果小于80%
            print("\nTrying alternative saving methods...")
            
            # 方法2: 使用_use_new_zipfile_serialization=False (PyTorch旧格式)
            alt_path1 = output_path.replace('.pth', '_alt1.pth')
            torch.save(save_dict, alt_path1, _use_new_zipfile_serialization=False)
            alt1_size = os.path.getsize(alt_path1) / (1024*1024)
            print(f"  Alternative 1 (old format): {alt1_size:.2f} MB")
            
            # 方法3: 使用pickle协议
            import pickle
            alt_path2 = output_path.replace('.pth', '_alt2.pth')
            with open(alt_path2, 'wb') as f:
                pickle.dump(save_dict, f, protocol=4)
            alt2_size = os.path.getsize(alt_path2) / (1024*1024)
            print(f"  Alternative 2 (pickle): {alt2_size:.2f} MB")
            
            # 选择最接近原始大小的文件
            sizes = {
                'default': file_size,
                'old_format': alt1_size,
                'pickle': alt2_size
            }
            best_method = min(sizes.keys(), key=lambda x: abs(sizes[x] - original_size))
            
            print(f"\n🎯 Best matching method: {best_method} ({sizes[best_method]:.2f} MB)")
            
            if best_method != 'default':
                # 替换为最佳版本
                best_path = output_path.replace('.pth', f'_{best_method}.pth')
                os.rename(best_path, output_path)
                print(f"Replaced with {best_method} version")
                
                # 删除其他临时文件
                for method, path in [('alt1', alt_path1), ('alt2', alt_path2)]:
                    if method != best_method and os.path.exists(path):
                        os.remove(path)
    
    print(f"\n✅ Averaged model saved: {output_path}")
    return True

def analyze_saved_model(model_path):
    """分析保存的模型文件"""
    print(f"\n🔍 Analyzing saved model: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
    
    file_size = os.path.getsize(model_path) / (1024*1024)
    print(f"File size: {file_size:.2f} MB")
    
    # 检查是否可以加载
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model can be loaded successfully")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"State dict keys: {len(state_dict)}")
            
            # 检查参数数量
            total_params = 0
            for key, tensor in state_dict.items():
                total_params += tensor.numel()
            print(f"Total parameters: {total_params:,}")
            
        if 'epochs_used' in checkpoint:
            print(f"Epochs used: {checkpoint['epochs_used']}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    # 配置路径
    checkpoints_dir = "/data/dusy/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0"
    output_path = "/data/dusy/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/averaged_model_final.pth"
    
    print("🧪 Starting checkpoint averaging with proper saving...")
    
    # 运行平均
    success = average_checkpoints_with_proper_saving(
        checkpoints_dir=checkpoints_dir,
        output_path=output_path,
        num_best=30
    )
    
    # 分析结果
    if success:
        analyze_saved_model(output_path)
        print("\n🎉 Process completed!")
        
        # 比较与原始checkpoint
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_epoch_*.pth"))
        if checkpoint_files:
            original_size = os.path.getsize(checkpoint_files[0]) / (1024*1024)
            averaged_size = os.path.getsize(output_path) / (1024*1024)
            print(f"\n📈 Final comparison:")
            print(f"  Original: {original_size:.2f} MB")
            print(f"  Averaged: {averaged_size:.2f} MB")
            print(f"  Ratio: {averaged_size/original_size:.2f}")
            
            # 重要提示
            if abs(averaged_size - original_size) / original_size < 0.1:
                print("✅ File sizes are well matched!")
            else:
                print("⚠️  File sizes differ, but this might be normal due to:")
                print("   - Different PyTorch versions")
                print("   - Different compression settings")
                print("   - Metadata differences")
                print("   The important thing is that parameter counts match and the model loads correctly.")
    else:
        print("\n❌ Averaging failed!")