

from copy import deepcopy

import torch
from captionvqa.common.registry import registry
from captionvqa.models.base_model import BaseModel
from captionvqa.modules.layers import ClassifierLayer, ConvNet, Flatten
from torch import nn


_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("cnn_lstm")
class CNNLSTM(BaseModel):
    """CNNLSTM is a simple model for vision and language tasks. CNNLSTM is supposed
    to acts as a baseline to test out your stuff without any complex functionality.
    Passes image through a CNN, and text through an LSTM and fuses them using
    concatenation. Then, it finally passes the fused representation from a MLP to
    generate scores for each of the possible answers.

    Args:
        config (DictConfig): Configuration node containing all of the necessary
                             config required to initialize CNNLSTM.

    Inputs: sample_list (SampleList)
        - **sample_list** should contain image attribute for image, text for
          question split into word indices, targets for answer scores
    """

    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/cnn_lstm/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0
        num_question_choices = registry.get(
            _TEMPLATES["question_vocab_size"].format(self._datasets[0])
        )
        num_answer_choices = registry.get(
            _TEMPLATES["number_of_answers"].format(self._datasets[0])
        )

        self.text_embedding = nn.Embedding(
            num_question_choices, self.config.text_embedding.embedding_dim
        )
        self.lstm = nn.LSTM(**self.config.lstm)

        layers_config = self.config.cnn.layers
        conv_layers = []
        for i in range(len(layers_config.input_dims)):
            conv_layers.append(
                ConvNet(
                    layers_config.input_dims[i],
                    layers_config.output_dims[i],
                    kernel_size=layers_config.kernel_sizes[i],
                )
            )
        conv_layers.append(Flatten())
        self.cnn = nn.Sequential(*conv_layers)

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = num_answer_choices
        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )

    def forward(self, sample_list):
        self.lstm.flatten_parameters()

        question = sample_list.text
        image = sample_list.image

        # Get (h_n, c_n), last hidden and cell state
        _, hidden = self.lstm(self.text_embedding(question))
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden = hidden[0].transpose(0, 1)

        # X should be 2 so we can merge in that dimension
        assert hidden.size(1) == 2, _CONSTANTS["hidden_state_warning"]

        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)
        image = self.cnn(image)

        # Fuse into single dimension
        fused = torch.cat([hidden, image], dim=-1)
        scores = self.classifier(fused)

        return {"scores": scores}
