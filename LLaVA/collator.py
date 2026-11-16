"""How a conversation about a plan is put to the language model: as a chat in the model's own template.

Human turns become the user's and gpt turns the assistant's. Where a turn says ``<image>`` stands one
placeholder token per patch, to be overwritten with the projected patches. With no system turn, Qwen's
template adds its default one. Only the assistant's answers and the ends of its turns are learned.
"""

import numpy as np
import torch

IMAGE_SLOT = "<|image_pad|>"
"""A Qwen vocabulary token that marks a patch"""
IGNORED = -100
"""The label the language model's loss passes over"""
ROLES = {"human": "user", "gpt": "assistant"}
"""TODO: Update role names in datasets"""


class ConversationCollator:
    """Turns plan conversations into a batch, right-padded to the longest one:

    - ``tokens`` ``(plans, tokens)``
    - ``attention_mask`` ``(plans, tokens)``, 0 over the padding
    - ``labels`` ``(plans, tokens)``, the input itself over the answers and ``IGNORED`` elsewhere
    - ``patches`` ``(plans, patches, embedding)``
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def get_tokens_and_labels(self, plan):
        """The plan's conversation as tokens with a label for each one."""
        
        messages, tokens, labels = [], [], []
        tokenized = ""
        for turn in plan.conversation:
            # Replace the image placeholder with one token per patch.
            message = {"role": ROLES[turn["from"]], "content": turn["value"].replace("<image>", IMAGE_SLOT * len(plan.patches))}
            if message["role"] == "assistant":
                # Format all the conversation so far, without tokenizing.
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Tokenize only the new part of the prompt and its answer.
                prompt_tokens = self.tokenize(prompt[len(tokenized):])
                answer_tokens = self.tokenize(message["content"] + self.tokenizer.eos_token)
                tokens += prompt_tokens + answer_tokens
                labels += [IGNORED] * len(prompt_tokens) + answer_tokens
                tokenized = prompt + message["content"] + self.tokenizer.eos_token
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
        batch["patches"] = torch.from_numpy(np.stack([plan.patches for plan in plans]))
        return batch
