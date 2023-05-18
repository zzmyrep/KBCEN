
import uuid
from typing import Optional

import pytorch_lightning as pl
from captionvqa.utils.build import build_dataloader_and_sampler
from captionvqa.utils.logger import log_class_usage
from omegaconf import DictConfig
from torch.utils.data import Dataset


# TODO(asg): Deprecate BaseDatasetBuilder after version release
class BaseDatasetBuilder(pl.LightningDataModule):
    def __init__(self, dataset_name: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if dataset_name is None:
            # In case user doesn't pass it
            dataset_name = f"dataset_{uuid.uuid4().hex[:6]}"
        self.dataset_name = dataset_name
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        log_class_usage("DatasetBuilder", self.__class__)

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    def prepare_data(self, config, *args, **kwargs):
        self.config = config
        self.build_dataset(config)

    def setup(self, stage: Optional[str] = None, config: Optional[DictConfig] = None):
        if config is None:
            config = self.config

        self.config = config
        self.train_dataset = self.load_dataset(config, "train")
        self.val_dataset = self.load_dataset(config, "val")
        self.test_dataset = self.load_dataset(config, "test")

    @property
    def train_dataset(self) -> Optional[Dataset]:
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset: Optional[Dataset]):
        self._train_dataset = dataset

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, dataset: Optional[Dataset]):
        self._val_dataset = dataset

    @property
    def test_dataset(self) -> Optional[Dataset]:
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset: Optional[Dataset]):
        self._test_dataset = dataset

    def build_dataset(self, config, dataset_type="train", *args, **kwargs):
        """
        Similar to load function, used by captionvqa to build a dataset for first
        time when it is not available. This internally calls 'build' function.
        Override that function in your child class.

        NOTE: The caller to this function should only call this on main process
        in a distributed settings so that downloads and build only happen
        on main process and others can just load it. Make sure to call
        synchronize afterwards to bring all processes in sync.

        Args:
            config (DictConfig): Configuration of this dataset loaded from
                                 config.
            dataset_type (str): Type of dataset, train|val|test

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``build``.
        """
        self.build(config, dataset_type, *args, **kwargs)

    def load_dataset(self, config, dataset_type="train", *args, **kwargs):
        """Main load function use by captionvqa. This will internally call ``load``
        function. Calls ``init_processors`` and ``try_fast_read`` on the
        dataset returned from ``load``

        Args:
            config (DictConfig): Configuration of this dataset loaded from config.
            dataset_type (str): Type of dataset, train|val|test

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``load``.
        """
        dataset = self.load(config, dataset_type, *args, **kwargs)
        if dataset is not None and hasattr(dataset, "init_processors"):
            # Checking for init_processors allows us to load some datasets
            # which don't have processors and don't inherit from BaseDataset
            dataset.init_processors()
        return dataset

    def load(self, config, dataset_type="train", *args, **kwargs):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.

        Args:
            config (DictConfig): Configuration of this dataset loaded from config.
            dataset_type (str): Type of dataset, train|val|test

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a load method"
        )

    @classmethod
    def config_path(cls):
        return None

    def build(self, config, dataset_type="train", *args, **kwargs):
        """
        This is used to build a dataset first time.
        Implement this method in your child dataset builder class.

        Args:
            config (DictConfig): Configuration of this dataset loaded from
                                 config.
            dataset_type (str): Type of dataset, train|val|test
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a build method"
        )

    def build_dataloader(
        self, dataset_instance: Optional[Dataset], dataset_type: str, *args, **kwargs
    ):
        if dataset_instance is None:
            raise TypeError(
                f"dataset instance for {dataset_type} hasn't been set and is None"
            )
        dataset_instance.dataset_type = dataset_type
        dataloader, _ = build_dataloader_and_sampler(dataset_instance, self.config)
        return dataloader

    def train_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.train_dataset, "train")

    def val_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.val_dataset, "val")

    def test_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.test_dataset, "test")

    def teardown(self, *args, **kwargs) -> None:
        pass
