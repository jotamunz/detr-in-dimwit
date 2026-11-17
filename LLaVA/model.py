"""A pretrained language model that reads a floor plan: the encoder's patches, projected into its token
embedding, take the places of the image slots in its prompt.
"""

import torch
from torch import nn


class Projector(nn.Module):
    """Maps each patch from the encoder's embedding into the language model's."""

    def __init__(self, patch_embedding, language_embedding):
        """
        :param patch_embedding: the width of an encoder patch
        :param language_embedding: the width of the language model's token embeddings
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(patch_embedding, language_embedding),
            nn.GELU(),
            nn.Linear(language_embedding, language_embedding),
        )

    def forward(self, patches):
        return self.layers(patches)


class LLaVA(nn.Module):
    """The frozen language model with a projector for the vision encoder."""

    def __init__(self, language_model, image_patch_id, patch_embedding):
        """
        :param language_model: the pretrained language model
        :param image_patch_id: the patch placeholder's token id in the language model's vocabulary
        :param patch_embedding: the width of an encoder patch
        """
        super().__init__()
        self.language_model = language_model.requires_grad_(False)
        self.image_patch_id = image_patch_id
        self.projector = Projector(patch_embedding, language_model.config.hidden_size)

    def forward(self, tokens, attention_mask, patches, labels=None):
        embeddings = self.language_model.get_input_embeddings()(tokens)
        # Keep the projector in float32 for its updates, whatever precision the language model runs in.
        projections = self.projector(patches.float()).to(embeddings.dtype)
        placeholders = tokens == self.image_patch_id
        replaced = embeddings.masked_scatter(placeholders.unsqueeze(-1), projections)
        return self.language_model(inputs_embeds=replaced, attention_mask=attention_mask, labels=labels)
