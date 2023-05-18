
import os
import warnings

from captionvqa.common.registry import registry
from captionvqa.datasets.builders.vqa2.builder import VQA2Builder
from captionvqa.datasets.builders.vqa2.masked_q_vqa2_dataset import MaskedQVQA2Dataset
from captionvqa.datasets.concat_dataset import captionvqaConcatDataset


@registry.register_builder("masked_q_vqa2")
class MaskedQVQA2Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_q_vqa2"
        self.dataset_class = MaskedQVQA2Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa2/masked_q.yaml"
