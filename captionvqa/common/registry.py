
from captionvqa.utils.env import setup_imports


class Registry:
    mapping = {
        # Use `registry.register_builder` to register a builder class
        # with a specific name
        # Further, use the name with the class is registered in the
        # command line or configuration to load that specific dataset
        "builder_name_mapping": {},
        # Similar to the builder_name_mapping above except that this
        # one is used to keep a mapping for dataset to its trainer class.
        "trainer_name_mapping": {},
        "model_name_mapping": {},
        "metric_name_mapping": {},
        "loss_name_mapping": {},
        "pool_name_mapping": {},
        "fusion_name_mapping": {},
        "optimizer_name_mapping": {},
        "scheduler_name_mapping": {},
        "processor_name_mapping": {},
        "encoder_name_mapping": {},
        "decoder_name_mapping": {},
        "transformer_backend_name_mapping": {},
        "transformer_head_name_mapping": {},
        "test_reporter_mapping": {},
        "iteration_strategy_name_mapping": {},
        "state": {},
        "callback_name_mapping": {},
    }

    @classmethod
    def register_trainer(cls, name):

        def wrap(trainer_cls):
            cls.mapping["trainer_name_mapping"][name] = trainer_cls
            return trainer_cls

        return wrap

    @classmethod
    def register_builder(cls, name):

        def wrap(builder_cls):
            from captionvqa.datasets.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class"
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def register_callback(cls, name):

        def wrap(func):
            from captionvqa.trainers.callbacks.base import Callback

            assert issubclass(
                func, Callback
            ), "All callbacks must inherit Callback class"
            cls.mapping["callback_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_metric(cls, name):

        def wrap(func):
            from captionvqa.modules.metrics import BaseMetric

            assert issubclass(
                func, BaseMetric
            ), "All Metric must inherit BaseMetric class"
            cls.mapping["metric_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_loss(cls, name):


        def wrap(func):
            from torch import nn

            assert issubclass(
                func, nn.Module
            ), "All loss must inherit torch.nn.Module class"
            cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_pooler(cls, name):


        def wrap(func):
            from torch import nn

            assert issubclass(
                func, nn.Module
            ), "All pooling methods must inherit torch.nn.Module class"
            cls.mapping["pool_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_fusion(cls, name):
        r"""Register a fusion technique to registry with key 'name'

        Args:
            name: Key with which the fusion technique will be registered

        Usage::

            from captionvqa.common.registry import registry
            from torch import nn

            @registry.register_fusion("linear_sum")
            class LinearSum():
                ...
        """

        def wrap(func):
            from torch import nn

            assert issubclass(
                func, nn.Module
            ), "All Fusion must inherit torch.nn.Module class"
            cls.mapping["fusion_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage::

            from captionvqa.common.registry import registry
            from captionvqa.models.base_model import BaseModel

            @registry.register_task("pythia")
            class Pythia(BaseModel):
                ...
        """

        def wrap(func):
            from captionvqa.models.base_model import BaseModel

            assert issubclass(
                func, BaseModel
            ), "All models must inherit BaseModel class"
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'

        Args:
            name: Key with which the processor will be registered.

        Usage::

            from captionvqa.common.registry import registry
            from captionvqa.datasets.processors import BaseProcessor

            @registry.register_task("glove")
            class GloVe(BaseProcessor):
                ...

        """

        def wrap(func):
            from captionvqa.datasets.processors.processors import BaseProcessor

            assert issubclass(
                func, BaseProcessor
            ), "All Processor classes must inherit BaseProcessor class"
            cls.mapping["processor_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_optimizer(cls, name):
        def wrap(func):
            cls.mapping["optimizer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_scheduler(cls, name):
        def wrap(func):
            cls.mapping["scheduler_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_transformer_backend(cls, name):
        def wrap(func):
            cls.mapping["transformer_backend_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_transformer_head(cls, name):
        def wrap(func):
            cls.mapping["transformer_head_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_test_reporter(cls, name):
        def wrap(func):
            cls.mapping["test_reporter_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_decoder(cls, name):
        r"""Register a decoder to registry with key 'name'

        Args:
            name: Key with which the decoder will be registered.

        Usage::

            from captionvqa.common.registry import registry
            from captionvqa.utils.text import TextDecoder


            @registry.register_decoder("nucleus_sampling")
            class NucleusSampling(TextDecoder):
                ...

        """

        def wrap(decoder_cls):
            from captionvqa.utils.text import TextDecoder

            assert issubclass(
                decoder_cls, TextDecoder
            ), "All decoders must inherit TextDecoder class"
            cls.mapping["decoder_name_mapping"][name] = decoder_cls
            return decoder_cls

        return wrap

    @classmethod
    def register_encoder(cls, name):
        r"""Register a encoder to registry with key 'name'

        Args:
            name: Key with which the encoder will be registered.

        Usage::

            from captionvqa.common.registry import registry
            from captionvqa.modules.encoders import Encoder


            @registry.register_encoder("transformer")
            class TransformerEncoder(Encoder):
                ...

        """

        def wrap(encoder_cls):
            from captionvqa.modules.encoders import Encoder

            assert issubclass(
                encoder_cls, Encoder
            ), "All encoders must inherit Encoder class"
            cls.mapping["encoder_name_mapping"][name] = encoder_cls
            return encoder_cls

        return wrap

    @classmethod
    def register_datamodule(cls, name):
        r"""Register a datamodule to registry with key 'name'

        Args:
            name: Key with which the datamodule will be registered.

        Usage::

            from captionvqa.common.registry import registry
            import pytorch_lightning as pl


            @registry.register_datamodule("my_datamodule")
            class MyDataModule(pl.LightningDataModule):
                ...

        """

        def wrap(datamodule_cls):
            import pytorch_lightning as pl

            assert issubclass(
                datamodule_cls, pl.LightningDataModule
            ), "All datamodules must inherit PyTorch Lightning DataModule class"
            cls.mapping["builder_name_mapping"][name] = datamodule_cls
            return datamodule_cls

        return wrap

    @classmethod
    def register_iteration_strategy(cls, name):
        r"""Register an iteration_strategy to registry with key 'name'

        Args:
            name: Key with which the iteration_strategy will be registered.

        Usage::

            from dataclasses import dataclass
            from captionvqa.common.registry import registry
            from captionvqa.datasets.iterators import IterationStrategy


            @registry.register_iteration_strategy("my_iteration_strategy")
            class MyStrategy(IterationStrategy):
                @dataclass
                class Config:
                    name: str = "my_strategy"
                def __init__(self, config, dataloader):
                    ...
        """

        def wrap(iteration_strategy_cls):
            from captionvqa.datasets.iteration_strategies import IterationStrategy

            assert issubclass(
                iteration_strategy_cls, IterationStrategy
            ), "All datamodules must inherit IterationStrategy class"
            cls.mapping["iteration_strategy_name_mapping"][
                name
            ] = iteration_strategy_cls
            return iteration_strategy_cls

        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from captionvqa.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_trainer_class(cls, name):
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_callback_class(cls, name):
        return cls.mapping["callback_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)
    # python中cls代表的是类的本身，https://www.cnblogs.com/wayne-tou/p/11896706.html

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def get_loss_class(cls, name):
        return cls.mapping["loss_name_mapping"].get(name, None)

    @classmethod
    def get_pool_class(cls, name):
        return cls.mapping["pool_name_mapping"].get(name, None)

    @classmethod
    def get_optimizer_class(cls, name):
        return cls.mapping["optimizer_name_mapping"].get(name, None)

    @classmethod
    def get_scheduler_class(cls, name):
        return cls.mapping["scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_decoder_class(cls, name):
        return cls.mapping["decoder_name_mapping"].get(name, None)

    @classmethod
    def get_encoder_class(cls, name):
        return cls.mapping["encoder_name_mapping"].get(name, None)

    @classmethod
    def get_iteration_strategy_class(cls, name):
        return cls.mapping["iteration_strategy_name_mapping"].get(name, None)

    @classmethod
    def get_transformer_backend_class(cls, name):
        return cls.mapping["transformer_backend_name_mapping"].get(name, None)

    @classmethod
    def get_transformer_head_class(cls, name):
        return cls.mapping["transformer_head_name_mapping"].get(name, None)

    @classmethod
    def get_test_rerporter_class(cls, name):
        return cls.mapping["test_reporter_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for captionvqa's
                               internal operations. Default: False
        Usage::

            from captionvqa.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from captionvqa.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()

# Only setup imports in main process, this means registry won't be
# fully available in spawned child processes (such as dataloader processes)
# but instantiated. This is to prevent issues such as
# https://github.com/None/captionvqa/issues/355
if __name__ == "__main__":
    setup_imports()
