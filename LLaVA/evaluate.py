"""Prints the answers a trained projector makes the language model write about the first plans of a
split, beside the dataset's own. Every answer is written from the conversation up to it as the dataset
has it, its earlier answers included, never from what the model wrote before.

The projector is the last one saved in its folder: the ``<update>.pt`` of the most updates.

    uv run caption.py --projector ../results/runs/<encoder run>/llava/<language model>/<run name>/patches-<dataset> \
        --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step>
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from collator import IMAGE_PATCH, format_conversation
from dataset import PlanConversation, PlanConversations
from model import LLaVA
from train import CAPTIONING

MAX_NEW_TOKENS = 80
"""Safety cutoff for generated text"""


@dataclass
class Turn:
    """Each assistant turn in a plan's conversation"""
    plan: PlanConversation
    """The plan in question"""
    question: str
    """The dataset's question"""
    prompt: str
    """The conversation up to the answer, in the chat template"""
    reference: str
    """The dataset's answer"""


def main():
    arguments = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arguments.add_argument("--projector", type=Path, required=True, help="the folder containing the trained projector's checkpoints")
    #TODO: Get embeddings automatically from the projector
    arguments.add_argument("--embeddings", type=Path, required=True, help="the folder containing the pregenerated embeddings")
    arguments.add_argument("--split", default="test")
    arguments.add_argument("--plans", type=int, default=50, help="how many plans")
    arguments.add_argument("--batch-size", type=int, default=32, help="answers written together")
    settings = arguments.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = max(settings.projector.glob("*.pt"), key=lambda path: int(path.stem))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("reading %s, a projector for %s" % (checkpoint_path, checkpoint["language_model"]))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["language_model"])
    language_model = AutoModelForCausalLM.from_pretrained(checkpoint["language_model"], dtype=torch.bfloat16)
    dataset = PlanConversations(settings.split, settings.embeddings, CAPTIONING)
    model = LLaVA(language_model, tokenizer.convert_tokens_to_ids(IMAGE_PATCH), patch_embedding=dataset[0].patches.shape[-1])
    model.projector.load_state_dict(checkpoint["projector"])
    model.to(device).eval()

    # Plain greedy decoding, in place of the sampling and repetition penalty the model ships with.
    greedy = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=language_model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    turns = []
    for plan in (dataset[index] for index in range(min(settings.plans, len(dataset)))):
        messages = format_conversation(plan)
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                prompt = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
                turns.append(Turn(plan, plan.conversation[index - 1]["value"], prompt, message["content"]))

    for start in range(0, len(turns), settings.batch_size):
        batch = turns[start:start + settings.batch_size]
        prompts = tokenizer([turn.prompt for turn in batch], padding=True, padding_side="left", add_special_tokens=False, return_tensors="pt")
        patches = torch.from_numpy(np.stack([turn.plan.patches for turn in batch]))
        written = model.generate(prompts["input_ids"].to(device), prompts["attention_mask"].to(device), patches.to(device), greedy)
        for turn, answer in zip(batch, tokenizer.batch_decode(written, skip_special_tokens=True)):
            print("\nplan %d | %s" % (turn.plan.seed, turn.question.replace("\n", " ")))
            print("  reference: %s" % turn.reference)
            print("  answered:   %s" % answer.strip())


if __name__ == "__main__":
    main()
