"""A pretrained language model that reads a floor plan: the elements d2g made of it, the patches, nodes or
relationships, each projected into the language model's token embedding, take the places of their
placeholders in its prompt.
"""

import torch
from torch import nn


class Projector(nn.Module):
    """Maps each element of a layer type from the transcribers embedding into the language model's."""

    def __init__(self, element_embedding, language_embedding):
        """
        :param element_embedding: the width of the transcribers' embeddings
        :param language_embedding: the width of the language model's token embeddings
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(element_embedding, language_embedding),
            nn.GELU(),
            nn.Linear(language_embedding, language_embedding),
        )

    def forward(self, elements):
        return self.layers(elements)


class LLaVA(nn.Module):
    """The language model with a projector for each layer type it reads."""

    def __init__(self, language_model, placeholders, element_embedding):
        """
        :param language_model: the pretrained language model
        :param placeholders: for each layer type, the token id in the language model's vocabulary that its elements replace
        :param element_embedding: the width of the transcribers' embeddings
        """
        super().__init__()
        self.language_model = language_model
        self.placeholders = placeholders
        self.projectors = nn.ModuleDict({layer_type: Projector(element_embedding, language_model.config.hidden_size) for layer_type in placeholders})

    def embed(self, tokens, elements):
        """The tokens' embeddings, with each layer type's projected elements in place of its placeholders."""
        embeddings = self.language_model.get_input_embeddings()(tokens)
        for layer_type, rows in elements.items():
            # Keep the projector in float32 for its updates, whatever precision the language model runs in.
            projections = self.projectors[layer_type](rows.float()).to(embeddings.dtype)
            placeholders = tokens == self.placeholders[layer_type]
            embeddings = embeddings.masked_scatter(placeholders.unsqueeze(-1), projections)
        return embeddings

    def forward(self, tokens, attention_mask, elements, labels=None):
        return self.language_model(inputs_embeds=self.embed(tokens, elements), attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, tokens, attention_mask, elements, generation_config):
        return self.language_model.generate(inputs_embeds=self.embed(tokens, elements), attention_mask=attention_mask, generation_config=generation_config)
