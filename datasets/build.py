import torch
from utils.transforms import build_transforms
from utils.utils import create_small_table
from .cuhk_sysu import CUHKSYSU
from .prw import PRW


def collate_fn(batch):
    return tuple(zip(*batch))


def print_statistics(dataset):
    """
    Print dataset statistics.
    """
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)
    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }
    if dataset.name != "CUHK-SYSU" and dataset.name != "CUHK-SYSU-COCO" or dataset.split != "query":
        pid_list = sorted(list(pid_set))
        if dataset.split == "query":
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                }
            )
        else:
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]  # remove unlabeled pid
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                    "unlabeled_pid": int(unlabeled_pid),
                }
            )
            # for train set, we need its num_train_pids for cluster init
            # unlabeled pid 5555
            dataset.num_train_pids = num_pids + 1
            # for train set(specifically target domain),we need num of boxes for init cluster
            dataset.num_boxes = num_boxes
    print(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))
    return dataset


class PersonSearchUDADataMoudle:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Train and Predict
        self.dataset_source_train = self.build_dataset(
            self.cfg.INPUT.DATASET, self.cfg.INPUT.DATA_ROOT, "train", is_source=True
        )
        self.dataset_target_train = self.build_dataset(
            self.cfg.INPUT.TDATASET, self.cfg.INPUT.TDATA_ROOT, "train", is_source=False
        )
        # Test
        self.gallery_set = self.build_dataset(self.cfg.INPUT.TDATASET, self.cfg.INPUT.TDATA_ROOT, "gallery")
        self.query_set = self.build_dataset(self.cfg.INPUT.TDATASET, self.cfg.INPUT.TDATA_ROOT, "query")

    def setup(self, stage):
        self.stage = stage

        if stage == "train":
            transforms_with_train = build_transforms(is_train=True)
            self.dataset_source_train.transforms = transforms_with_train
            self.dataset_target_train.transforms = transforms_with_train

        if stage == "test":
            transforms_without_train = build_transforms(is_train=False)
            self.gallery_set.transforms = transforms_without_train
            self.query_set.transforms = transforms_without_train

        if stage == "predict":
            transforms_without_train = build_transforms(is_train=False)
            self.dataset_source_train.transforms = transforms_without_train
            self.dataset_target_train.transforms = transforms_without_train

    def set_test_gallery(self, gallery_size):
        if isinstance(self.gallery_set, CUHKSYSU) and gallery_size in [50, 100, 500, 1000, 2000, 4000]:
            self.gallery_set.test_gallery_size = gallery_size
        else:
            print(f"You cannot set test gallery_size {gallery_size} at {self.gallery_set.__class__.__name__} dataset")

    def train_dataloader(self):
        train_loader_source = torch.utils.data.DataLoader(
            self.dataset_source_train,
            batch_size=self.cfg.INPUT.BATCH_SIZE_TRAIN,
            shuffle=True,
            num_workers=self.cfg.INPUT.NUM_WORKERS_TRAIN,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        train_loader_target = torch.utils.data.DataLoader(
            self.dataset_target_train,
            batch_size=self.cfg.INPUT.BATCH_SIZE_TRAIN,
            shuffle=True,
            num_workers=self.cfg.INPUT.NUM_WORKERS_TRAIN,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        return train_loader_source, train_loader_target

    def test_dataloader(self):
        gallery_loader = torch.utils.data.DataLoader(
            self.gallery_set,
            batch_size=self.cfg.INPUT.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.INPUT.NUM_WORKERS_TEST,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        query_loader = torch.utils.data.DataLoader(
            self.query_set,
            batch_size=self.cfg.INPUT.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.INPUT.NUM_WORKERS_TEST,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return gallery_loader, query_loader

    def predict_dataloader(self, is_source=True):
        if is_source:
            train_loader_source = torch.utils.data.DataLoader(
                self.dataset_source_train,
                batch_size=self.cfg.INPUT.BATCH_SIZE_TEST,
                shuffle=False,
                num_workers=self.cfg.INPUT.NUM_WORKERS_TEST,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            return train_loader_source
        else:
            train_loader_target = torch.utils.data.DataLoader(
                self.dataset_target_train,
                batch_size=self.cfg.INPUT.BATCH_SIZE_TEST,
                shuffle=False,
                num_workers=self.cfg.INPUT.NUM_WORKERS_TEST,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            return train_loader_target

    def build_dataset(self, dataset_name, root, split, is_source=True, build_tiny=False):
        if dataset_name == "CUHK-SYSU":
            dataset = CUHKSYSU(root, split, is_source=is_source, build_tiny=build_tiny)
        elif dataset_name == "PRW":
            dataset = PRW(root, split, is_source=is_source, build_tiny=build_tiny)

        dataset = print_statistics(dataset)
        return dataset

    def reset_dataset_with_pseudo_labels(self, img_proposal_boxes, sorted_keys, pseudo_labels):
        for i, anno in enumerate(self.dataset_target_train.annotations):
            boxes_nums = len(img_proposal_boxes[anno["img_name"]])
            anno["pids"] = torch.zeros(boxes_nums)
            anno["boxes"] = img_proposal_boxes[anno["img_name"]]
            for j in range(boxes_nums):
                index = sorted_keys.index(anno["img_name"] + "_" + str(j))
                label = pseudo_labels[index]
                anno["pids"][j] = label
            self.dataset_target_train.annotations[i] = anno
