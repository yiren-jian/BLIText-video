"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
# from PIL import Image
import numpy as np   ### Vatex I3D features are stored as npy
import torch


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["video"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class MyVatexCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.vis_processor = None    ### No visual processor for Vatex features

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["video"]).replace(".mp4", ".npy")
        # image = Image.open(image_path).convert("RGB")
        image = np.load(image_path).squeeze(0)   ### 1 x 32 x 1024 --> 32 x 1024
        image = torch.from_numpy(image).float()
        image_padded = torch.zeros(32, image.shape[1])
        image_padded[:image.shape[0]] = image

        mask = torch.zeros(32)
        mask[:image.shape[0]] = 1
        mask = mask.long()

        # image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image_padded,
            "mask": mask,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "image_name": ann["video"]
        }


class MyVatexCaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_names = []
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_names:
                self.img_names.append(img_id)

        self.vis_processor = None    ### No visual processor for Vatex features

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        img_name = self.img_names[index]

        image_path = os.path.join(self.vis_root, img_name).replace(".mp4", ".npy")
        image = np.load(image_path).squeeze(0)   ### 1 x 32 x 1024 --> 32 x 1024
        image = torch.from_numpy(image).float()
        image_padded = torch.zeros(32, image.shape[1])
        image_padded[:image.shape[0]] = image

        mask = torch.zeros(32)
        mask[:image.shape[0]] = 1
        mask = mask.long()

        return {
            "image": image_padded,
            "mask": mask,
            "image_id": img_name,
            "image_name": img_name
        }
