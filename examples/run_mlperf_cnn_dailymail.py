import argparse
import json
import logging
import os
import torch

from lm_eval import tasks, evaluator, utils, models
from typing import Optional, Union, List

import transformers
from transformers import BatchEncoding

logging.getLogger("openai").setLevel(logging.WARNING)

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

MAX_LENGTH=1919

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--batch_size", type=str, default=1)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")

    return parser.parse_args()

class CNNDaily(models.huggingface.AutoCausalLM):
    MAX_LENGTH=1919

    def init(
        self,
        pretrained: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[Union[int, str]] = "cuda",  
        ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True,
            model_max_length=MAX_LENGTH,
            padding_side="left",
            use_fast=False,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tok_encode(self, string: str):
        source_encoded = self.tokenizer(string, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        self.attention_mask = source_encoded.attention_mask
 
        return source_encoded.input_ids[0].tolist()

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            return_tensors="pt",
            padding="max_length",
            pad_to_max_length=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

    @property
    def max_length(self) -> int:
        return MAX_LENGTH
 
    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        ):
 
        generation_kwargs = {
                "early_stopping": True,
                "max_new_tokens": 128,
                "min_new_tokens": 30,
                "num_beams": 4,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                }

        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
        attention_mask = inputs["attention_mask"][
            :, self.max_gen_toks - self.max_length :
        ]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = models.huggingface.stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            **generation_kwargs,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        outputs = self.tok_decode(generations.tolist())
        print("generations: {}\n".format(outputs))

        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=input_ids.size(1)
        )


def main():
    args = parse_args()

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")
    assert task_names == ['cnn_dailymail']
    task_dict = tasks.get_task_dict(task_names)

    model = CNNDaily(args.model,device=args.device, batch_size=args.batch_size, max_gen_toks=128)
    results = evaluator.evaluate(
        model,
        task_dict,
        limit=args.limit
    )

    dumped = json.dumps(results, indent=2)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
