""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2019, Ross Wightman
"""
import random
from functools import partial
from typing import Callable

import torch.utils.data
import numpy as np

from transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)
    这是一个用于快速整理样本批次的函数，针对uint8类型的图像和int64类型的标签。根据输入数据的类型进行不同的处理，最终返回整理好的图像数据和标签。
    fast_collate函数利用PyTorch中的C++扩展来加速样本批次的组合过程。它能够更高效地处理多维数据，例如图像数据。通过使用fast_collate函数，可以减少数据加载的时间，提高数据加载器的性能。
    fast_collate函数并没有直接使用C++扩展，而是在底层使用了一些PyTorch的内置函数和操作，这些函数和操作是使用C++实现的，以提供更高效的数据处理和组合。以下是一些可能在fast_collate函数内部使用的相关C++扩展和操作：
    torch.stack: 这是一个用于沿新维度组合张量序列的操作，它在底层使用C++实现以提高效率。
    torch.as_tensor: 这是一个用于将Python对象转换为张量的函数，它底层使用C++实现。
    torch.from_numpy: 这是一个将NumPy数组转换为张量的函数，它在底层使用C++实现。
    torch.Tensor类和相关操作：PyTorch中的Tensor类及其相关操作在底层使用C++实现，包括张量的创建、形状变换、拼接等操作。"""
    #断言batch[0]是一个元组，表示输入的batch应该是一个元组列表。
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


class PrefetchLoader:
    ''' 当use_prefetcher为True时，会在创建数据加载器时使用预取加载器。预取加载器会异步地预取和处理下一个批次的数据，以减少数据加载和预处理的等待时间，并且可以在 GPU 计算的同时进行数据准备，提高整体的训练速度。
    预取加载器在训练过程中对数据进行了额外的处理，例如图像均值标准化、随机擦除等，以提供更多的数据增强功能和数据预处理选项。
    如果use_prefetcher为False，则不使用预取加载器，将使用普通的torch.utils.data.DataLoader来加载数据。
    但也需要更多的内存和计算资源来支持预取和预处理操作。因此，根据具体的训练场景和硬件条件，可以选择是否使用预取加载器。'''
    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x


def _worker_init(worker_id, worker_seeding='all'):
    '''这是一个用于初始化数据加载器的函数，在多线程环境下设置随机种子，以确保数据加载的可重复性。'''
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
):
    '''这是一个创建数据加载器的函数，它接收一系列参数，包括数据集、输入大小、批处理大小等。根据参数配置，创建一个数据加载器，并根据需要添加预处理功能，如数据增强、随机擦除等。
    re_prob参数表示随机擦除（Random Erasing）的概率。随机擦除是一种数据增强技术，在训练过程中随机选择图像的一部分区域并将其擦除，然后用随机的像素值进行填充。这样可以强制模型学习对遮挡和缺失信息的鲁棒性，从而提高模型的泛化能力。'''
    re_num_splits = 0
    if re_split: #即是否需要将批次分割为两部分进行处理。
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    #调用create_transform函数创建数据集的转换操作，例如是否进行数据增强（no_aug参数）、缩放和裁剪的比例（scale和ratio参数）、是否进行水平翻转（hflip参数）等。
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    '''这部分代码根据参数distributed的值判断是否启用分布式训练，并根据不同的情况创建采样器（sampler）。
    如果启用分布式训练且数据集不是torch.utils.data.IterableDataset类型，则根据训练模式和num_aug_repeats参数创建不同的采样器。
    如果是分布式训练且处于训练模式，且num_aug_repeats不为零，则创建RepeatAugSampler采样器，该采样器可用于重复增强数据。否则，创建DistributedSampler采样器，用于在分布式环境下对数据集进行划分。
    如果不是分布式训练，或者是分布式训练但数据集是torch.utils.data.IterableDataset类型，则断言num_aug_repeats为零，因为在非分布式或IterableDataset的情况下不支持重复增强。'''
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
    #collate_fn用于将单个样本组合成一个批次。
    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate
    #
    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    '''这段代码设置数据加载器的参数，使用字典loader_args存储这些参数。
    参数包括批处理大小（batch_size）、是否打乱数据（根据条件判断）、工作进程数量（num_workers）、采样器（sampler）、collate_fn、是否将数据加载到固定的内存中（pin_memory）、
    是否丢弃最后一个不完整的批次（根据训练模式判断）、工作进程初始化函数（worker_init_fn）、是否使用持久化工作进程（persistent_workers）'''
    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    #将原始的数据加载器loader包装在PrefetchLoader中，以加速数据加载，并提供额外的功能，如图像均值标准化、随机擦除等。
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    ''' 这是一个支持多个epoch训练的数据加载器类，它基于原始的DataLoader类实现。通过重复迭代训练数据集，实现多个周期的训练。'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
         这是一个无限循环的采样器类，用于在多个周期训练中重复采样数据。

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
