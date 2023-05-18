
from captionvqa.common.registry import registry
from captionvqa.datasets.builders.okvqa.dataset import OKVQADataset
from captionvqa.datasets.captionvqa_dataset_builder import captionvqaDatasetBuilder


@registry.register_builder("okvqa")
class OKVQABuilder(captionvqaDatasetBuilder):
    def __init__(
        self, dataset_name="okvqa", dataset_class=OKVQADataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/okvqa/defaults.yaml"
