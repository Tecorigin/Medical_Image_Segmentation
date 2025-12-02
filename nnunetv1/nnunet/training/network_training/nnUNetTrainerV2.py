#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple
import os
import numpy as np
import torch
import torch_sdaa
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

device = torch.device('sdaa')

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 180  # 修改为180epoch
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        
        # 新增：用于保存验证loss和对应的checkpoint
        self.val_loss_history = []  # 存储每个epoch的验证loss
        self.best_checkpoints = []  # 存储最佳checkpoint信息
        self.max_best_checkpoints = 30  # 保存最佳30个checkpoint

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.sdaa.is_available():
            self.network.sdaa()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        scaler = torch.sdaa.amp.GradScaler()

        if torch.sdaa.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                # print('fp16')
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                # print('fp16 backprop')
                scaler.scale(l).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            # print('fp32')
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                # print('fp32 backprop')
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        self.print_to_log_file("Step loss: %.4f" % l.detach().cpu().numpy())
        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    # def on_epoch_end(self):
    #     """
    #     修改：每个epoch结束后进行验证，保存checkpoint并记录验证loss
    #     :return:
    #     """
    #     super().on_epoch_end()
        
    #     # 每个epoch都进行验证
    #     val_loss = self.validate_epoch()
    #     self.val_loss_history.append((self.epoch, val_loss))
        
    #     # 保存当前epoch的checkpoint
    #     self.save_checkpoint()
        
    #     # 更新最佳checkpoint列表
    #     self.update_best_checkpoints()
        
    #     # 记录验证loss到日志
    #     self.print_to_log_file(f"Epoch {self.epoch} - Validation Loss: {val_loss:.4f}")
        
    #     continue_training = self.epoch < self.max_num_epochs

    #     # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
    #     # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
    #     if self.epoch == 100:
    #         if self.all_val_eval_metrics[-1] == 0:
    #             self.optimizer.param_groups[0]["momentum"] = 0.95
    #             self.network.apply(InitWeights_He(1e-2))
    #             self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
    #                                    "high momentum. High momentum (0.99) is good for datasets where it works, but "
    #                                    "sometimes causes issues such as this one. Momentum has now been reduced to "
    #                                    "0.95 and network weights have been reinitialized")
    #     return continue_training

    def on_epoch_end(self):
        """
        修改：每个epoch结束后进行验证，保存checkpoint并记录验证loss
        :return:
        """
        # 先调用父类的on_epoch_end
        continue_training = super().on_epoch_end()
        
        # 从父类验证结果中获取loss
        if hasattr(self, 'all_val_losses') and len(self.all_val_losses) > 0:
            val_loss = self.all_val_losses[-1]
        elif hasattr(self, 'all_val_eval_metrics') and len(self.all_val_eval_metrics) > 0:
            # 如果没有loss记录，使用Dice指标的倒数作为loss的替代
            # Dice越高越好，所以我们用 1 - Dice 作为loss
            avg_dice = np.mean(self.all_val_eval_metrics[-1])
            val_loss = 1.0 - avg_dice
        else:
            val_loss = float('inf')
        
        self.val_loss_history.append((self.epoch, val_loss))
        
        # 保存当前epoch的checkpoint（使用默认命名）
        self.save_checkpoint()
        
        # 更新最佳checkpoint列表
        self.update_best_checkpoints()
        
        # 记录验证loss到日志
        self.print_to_log_file(f"Epoch {self.epoch} - Validation Loss: {val_loss:.4f}")
        
        # 检查是否继续训练
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def save_checkpoint(self, checkpoint_file=None):
        """
        新增：保存当前epoch的checkpoint
        :param checkpoint_file: 如果为None，使用默认命名；否则使用指定文件名
        """
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': self.val_loss_history[-1][1] if self.val_loss_history else float('inf')
        }
        
        if checkpoint_file is None:
            # 使用默认命名
            checkpoint_file = join(self.output_folder, f'checkpoint_epoch_{self.epoch:03d}.pth')
        else:
            # 使用指定的文件名
            checkpoint_file = join(self.output_folder, checkpoint_file)
        
        torch.save(checkpoint, checkpoint_file)
        self.print_to_log_file(f"Saved checkpoint: {checkpoint_file}")


    # def validate_epoch(self):
    #     """
    #     新增：计算当前epoch的验证loss
    #     :return: 验证loss
    #     """
    #     self.network.eval()
    #     total_val_loss = 0.0
    #     num_batches = 0
        
    #     with torch.no_grad():
    #         for data_dict in self.dl_val:
    #             # 检查数据字典中的键名
    #             if 'target' in data_dict:
    #                 data = data_dict['data']
    #                 target = data_dict['target']
    #             elif 'seg' in data_dict:
    #                 # 有些数据生成器使用 'seg' 作为标签键
    #                 data = data_dict['data']
    #                 target = data_dict['seg']
    #             else:
    #                 # 如果找不到目标键，尝试使用第一个非数据键
    #                 data = data_dict['data']
    #                 # 找到第一个不是'data'的键作为目标
    #                 target_key = [key for key in data_dict.keys() if key != 'data'][0]
    #                 target = data_dict[target_key]
    #                 self.print_to_log_file(f"Warning: Using '{target_key}' as target key")
                
    #             data = maybe_to_torch(data)
    #             target = maybe_to_torch(target)
                
    #             if torch.sdaa.is_available():
    #                 data = to_cuda(data)
    #                 target = to_cuda(target)
                
    #             output = self.network(data)
    #             loss = self.loss(output, target)
                
    #             total_val_loss += loss.item()
    #             num_batches += 1
        
    #     avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    #     return avg_val_loss


    def update_best_checkpoints(self):
        """
        新增：更新最佳checkpoint列表（保留验证loss最小的30个）
        """
        if not self.val_loss_history:
            return
            
        current_epoch, current_val_loss = self.val_loss_history[-1]
        
        # 添加当前checkpoint到列表
        self.best_checkpoints.append((current_epoch, current_val_loss, 
                                     join(self.output_folder, f'checkpoint_epoch_{current_epoch:03d}.pth')))
        
        # 按验证loss排序并保留前30个
        self.best_checkpoints.sort(key=lambda x: x[1])
        if len(self.best_checkpoints) > self.max_best_checkpoints:
            # 删除多余的checkpoint文件
            for epoch, _, checkpoint_path in self.best_checkpoints[self.max_best_checkpoints:]:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    self.print_to_log_file(f"Removed checkpoint: {checkpoint_path}")
            
            # 保留前30个
            self.best_checkpoints = self.best_checkpoints[:self.max_best_checkpoints]
        
        # 记录当前最佳checkpoint信息
        self.print_to_log_file(f"Current best checkpoints (top {min(len(self.best_checkpoints), 5)}):")
        for i, (epoch, loss, _) in enumerate(self.best_checkpoints[:5]):
            self.print_to_log_file(f"  {i+1}. Epoch {epoch}: {loss:.4f}")

    def create_averaged_model(self):
        """
        新增：创建平均模型（使用最佳30个checkpoint的平均权重）
        :return: 平均模型
        """
        if len(self.best_checkpoints) == 0:
            self.print_to_log_file("No best checkpoints found for averaging!")
            return None
        
        self.print_to_log_file(f"Creating averaged model from {len(self.best_checkpoints)} best checkpoints...")
        
        # 初始化平均权重
        averaged_state_dict = None
        
        for i, (epoch, val_loss, checkpoint_path) in enumerate(self.best_checkpoints):
            if not os.path.exists(checkpoint_path):
                self.print_to_log_file(f"Checkpoint not found: {checkpoint_path}")
                continue
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            if averaged_state_dict is None:
                averaged_state_dict = {key: value.clone() for key, value in state_dict.items()}
            else:
                for key in averaged_state_dict:
                    averaged_state_dict[key] += state_dict[key]
        
        # 计算平均
        for key in averaged_state_dict:
            averaged_state_dict[key] = averaged_state_dict[key] / len(self.best_checkpoints)
        
        # 创建新模型并加载平均权重
        averaged_model = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                     len(self.net_num_pool_op_kernel_sizes),
                                     self.conv_per_stage, 2, 
                                     nn.Conv3d if self.threeD else nn.Conv2d,
                                     nn.InstanceNorm3d if self.threeD else nn.InstanceNorm2d,
                                     {'eps': 1e-5, 'affine': True},
                                     nn.Dropout3d if self.threeD else nn.Dropout2d,
                                     {'p': 0, 'inplace': True},
                                     nn.LeakyReLU,
                                     {'negative_slope': 1e-2, 'inplace': True},
                                     True, False, lambda x: x, InitWeights_He(1e-2),
                                     self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        
        averaged_model.load_state_dict(averaged_state_dict)
        averaged_model.inference_apply_nonlin = softmax_helper
        
        if torch.sdaa.is_available():
            averaged_model.sdaa()
        
        self.print_to_log_file("Averaged model created successfully!")
        return averaged_model

    def run_training(self):
        """
        修改：训练结束后创建平均模型并在测试集上测试
        :return:
        """
        self.maybe_update_lr(self.epoch)
        ds = self.network.do_ds
        self.network.do_ds = True
        
        # 运行原始训练
        ret = super().run_training()
        self.network.do_ds = ds
        
        # 训练结束后创建平均模型
        if self.epoch >= self.max_num_epochs:
            self.print_to_log_file("Training completed. Creating averaged model...")
            averaged_model = self.create_averaged_model()
            
            if averaged_model is not None:
                # 保存平均模型
                averaged_model_path = join(self.output_folder, 'averaged_model.pth')
                torch.save({
                    'state_dict': averaged_model.state_dict(),
                    'epochs_used': [epoch for epoch, _, _ in self.best_checkpoints],
                    'val_losses': [loss for _, loss, _ in self.best_checkpoints]
                }, averaged_model_path)
                self.print_to_log_file(f"Averaged model saved: {averaged_model_path}")
                
                # 在测试集上测试平均模型
                self.test_averaged_model(averaged_model)
        
        return ret

    def test_averaged_model(self, averaged_model):
        """
        新增：在测试集上测试平均模型
        :param averaged_model: 平均模型
        """
        self.print_to_log_file("Testing averaged model on test set...")
        
        # 备份原始网络
        original_network = self.network
        # 使用平均模型进行测试
        self.network = averaged_model
        
        try:
            # 运行测试
            self.validate(do_mirroring=True, use_sliding_window=True, 
                         validation_folder_name='validation_averaged')
            self.print_to_log_file("Averaged model testing completed!")
        except Exception as e:
            self.print_to_log_file(f"Error during averaged model testing: {e}")
        finally:
            # 恢复原始网络
            self.network = original_network