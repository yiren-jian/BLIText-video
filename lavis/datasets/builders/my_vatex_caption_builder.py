"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry

from lavis.datasets.datasets.my_vatex_caption_datasets import (
    MyVatexCaptionDataset,
    MyVatexCaptionEvalDataset,
)

@registry.register_builder("my_vatex_caption")
class MyVATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = MyVatexCaptionDataset
    eval_dataset_cls = MyVatexCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/my_vatex_cap.yaml",
    }
