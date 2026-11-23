"""How a conversation about a plan is put to the language model: as a chat in the model's own template.

Human turns become the user's and gpt turns the assistant's. Where a turn says ``<image>`` stands one
placeholder token per patch, and where it says ``<record>`` one line per node and then per relationship,
its name and a placeholder token, ``E0: <|vision_pad|>`` or ``R0: <|video_pad|>``, so that a question
about ``E19`` finds its node. The placeholders are overwritten with the projected elements. With no system
turn, Qwen's template adds its default one. Only the assistant's answers and the ends of its turns are
learned.
"""

import numpy as np
import torch

PLACEHOLDERS = {"patches": "<|image_pad|>", "nodes": "<|vision_pad|>", "relationships": "<|video_pad|>"}
"""For each layer type, the Qwen vocabulary token that marks one of its elements"""
ENTITY_NAMES = {"nodes": "E", "relationships": "R"}
"""How the conversations refer to a node or relationship"""
IGNORED = -100
"""The label the language model's loss passes over"""
ROLES = {"human": "user", "gpt": "assistant"}
# TODO: Update role names in datasets


def format_conversation(plan):
    image = PLACEHOLDERS["patches"] * len(plan.elements.get("patches", []))
    record = "\n".join(
        "%s%d: %s" % (ENTITY_NAMES[layer_type], number, PLACEHOLDERS[layer_type])
        for layer_type in ENTITY_NAMES if layer_type in plan.elements
        for number in range(len(plan.elements[layer_type]))
    )
    return [
        {"role": ROLES[turn["from"]], "content": turn["value"].replace("<image>", image).replace("<record>", record)}
        for turn in plan.conversation
    ]


class ConversationCollator:
    """Turns plan conversations into a batch, right-padded to the longest one:

    - ``tokens`` ``(plans, tokens)``
    - ``attention_mask`` ``(plans, tokens)``, 0 over the padding
    - ``labels`` ``(plans, tokens)``, the input itself over the answers and ``IGNORED`` elsewhere
    - ``elements`` for each layer type, ``(elements, embedding)`` the embedding of each element in the plan
    """

    def __init__(self, tokenizer):
        """
        :param tokenizer: the language model's tokenizer
        """
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def get_tokens_and_labels(self, plan):
        """The plan's conversation as tokens with a label for each one."""

        messages, tokens, labels = [], [], []
        tokenized_text = ""
        for message in format_conversation(plan):
            if message["role"] == "assistant":
                # Format all the conversation so far, without tokenizing.
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Tokenize only the new part of the prompt and its answer.
                prompt_tokens = self.tokenize(prompt[len(tokenized_text):])
                answer_tokens = self.tokenize(message["content"] + self.tokenizer.eos_token)
                tokens += prompt_tokens + answer_tokens
                labels += [IGNORED] * len(prompt_tokens) + answer_tokens
                tokenized_text = prompt + message["content"] + self.tokenizer.eos_token
            messages.append(message)
        return tokens, labels

    def __call__(self, plans):
        conversations = [self.get_tokens_and_labels(plan) for plan in plans]
        longest = max(len(tokens) for tokens, _ in conversations)
        batch = {"tokens": [], "attention_mask": [], "labels": []}
        for tokens, labels in conversations:
            padding = longest - len(tokens)
            batch["tokens"].append(tokens + [self.tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(tokens) + [0] * padding)
            batch["labels"].append(labels + [IGNORED] * padding)

        batch = {key: torch.tensor(rows) for key, rows in batch.items()}
        batch["elements"] = {layer_type: torch.from_numpy(np.concatenate([plan.elements[layer_type] for plan in plans])) for layer_type in plans[0].elements}
        return batch
