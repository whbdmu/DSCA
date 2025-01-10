import glob
import re
import os.path as osp

from ..utils.data import BaseDataset


class VehicleX(BaseDataset):
    """
    VeRi
    Reference:
    PAMTRI: Pose-Aware Multi-Task Learning for Vehicle Re-Identification Using Highly Randomized Synthetic Data. In: ICCV 2019
    """

    dataset_dir = "AIC20_ReID_Simulation"

    def __init__(self, root, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "image_train")

        self.check_before_run()

        train = self.process_dir(self.train_dir, relabel=True)

        if verbose:
            print("=> VehicleX loaded")
            self.print_dataset_statistics(train)

        self.train = train

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f'"{self.dataset_dir}" is not available')
        if not osp.exists(self.train_dir):
            raise RuntimeError(f'"{self.train_dir}" is not available')

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c([-\d]+)")

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= pid <= 1362
            assert 6 <= camid <= 36
            camid -= 6  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset

    def print_dataset_statistics(self, train):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print(f"  train    | {num_train_pids:5d} | {num_train_imgs:8d} | {num_train_cams:9d}")
        print("  ----------------------------------------")
