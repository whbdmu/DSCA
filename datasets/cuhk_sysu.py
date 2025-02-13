import os.path as osp

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset
from collections import OrderedDict


class CUHKSYSU(BaseDataset):
    def __init__(self, root, split, is_source=True, build_tiny=False, test_gallery_size=50):
        self.name = "CUHK-SYSU"
        self.img_prefix = osp.join(root, "Image", "SSM")
        self.test_gallery_size = test_gallery_size
        self.test_set_config = {
            gallery_size: (f"annotation/test/train_test/TestG{gallery_size}.mat", f"TestG{gallery_size}")
            for gallery_size in [50, 100, 500, 1000, 2000, 4000]  # These test file have a same query sequence
        }

        super().__init__(root, split, is_source=is_source, build_tiny=build_tiny)

    def _load_queries(self):
        # TestG50: a test protocol, 50 gallery images per query
        path, key = self.test_set_config[self.test_gallery_size]
        protoc = loadmat(osp.join(self.root, path))[key].squeeze()
        queries = []
        for item in protoc["Query"]:
            img_name = str(item["imname"][0, 0][0])
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([-100]),  # dummy pid
                }
            )
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        # gallery images
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs]
        if self.split == "gallery":
            return gallery_imgs
        # all images
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training images = all images - gallery images
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        # load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        unlabeled_pid = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:
            img_name = str(img_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, f"Warning: {img_name} has no valid boxes."
            boxes = boxes[valid_index]
            name_to_boxes[img_name] = boxes.astype(np.int32)
            name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return

        # assign a unique pid from 1 to N for each identity
        if self.split == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for img_name, box, _ in scenes:
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
        else:
            path, key = self.test_set_config[self.test_gallery_size]
            protoc = loadmat(osp.join(self.root, path))[key].squeeze()
            for index, item in enumerate(protoc):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

        annotations = []
        imgs = self._load_split_img_names()

        # build a subset of 500 images for debugging, note that we need to rearrange the pid labels
        # 用500张作subset来debug, 需要重新分配pid
        if self.build_tiny:
            imgs = imgs[:500]
            exist_pids = OrderedDict()
        for img_name in imgs:
            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
            pids = name_to_pids[img_name]
            if self.build_tiny:
                for id in pids:
                    exist_pids[id] = id
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": boxes,
                    "pids": pids,
                }
            )
        if self.build_tiny:
            # debug用tiny时重新分配pid
            for i, (key, value) in enumerate(exist_pids.items()):
                if key != unlabeled_pid:
                    exist_pids[key] = i + 1
            for anno in annotations:
                for index in range(len(anno["pids"])):
                    anno["pids"][index] = exist_pids[anno["pids"][index]]
        return annotations
