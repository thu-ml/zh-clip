# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch ZH-CLIP model."""


from typing import Optional, Tuple, Union
from torch import TensorType

import torch
from torch import nn


from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput
from transformers.models.auto.modeling_auto import AutoModel

from transformers.models.clip.modeling_clip import CLIPVisionConfig, CLIPVisionModel
from .configuration_zhclip import ZhCLIPConfig
from dataclasses import dataclass

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "ZhCLIPConfig"

@dataclass
class ZhCLIPModelOutput(ModelOutput):

    text_features: torch.FloatTensor = None
    image_features: torch.FloatTensor = None


class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, last_hidden_state: TensorType, attention_mask: TensorType):
        masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


class ZhCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = ZhCLIPConfig
    base_model_prefix = "zhclip"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ZhCLIPModel(ZhCLIPPreTrainedModel):
    def __init__(
        self,
        config: Optional[ZhCLIPConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):

        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            config = ZhCLIPConfig(vision_model.config, text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config).vision_model
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        self.vision_model = vision_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.coattention_dim = config.hidden_size

        # add projection layers
        mlp_hidden_size = (self.text_embed_dim + self.coattention_dim) // 2
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, mlp_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, self.coattention_dim, bias=False),
        )
        self.text_pooler = MeanPooler()
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.coattention_dim)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        patch_ids = None,
        extend_token_type_ids = None,
        return_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ZhCLIPModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            return_dict=return_dict,
        )
        text_features = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )
        return ZhCLIPModelOutput(
            image_features = image_features,
            text_features = text_features,
        )


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)


    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        text_pool = self.text_pooler(text_outputs[0], attention_mask)
        text_feat = self.text_projection(text_pool)
        return text_feat


    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
