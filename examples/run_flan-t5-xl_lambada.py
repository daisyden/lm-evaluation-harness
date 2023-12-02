import argparse
import json
import logging
import os
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval import tasks, evaluator, utils, models
from typing import Optional, Union, List, Tuple

import transformers
from transformers import BatchEncoding

logging.getLogger("openai").setLevel(logging.WARNING)

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

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

class T5ModelLambada(models.huggingface.AutoSeq2SeqLM):
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                if length >= 1 and target_tokens[length-1].item() == self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens = False)[0]:
                    length = length - 1

                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_text = self.tokenizer.decode(target_tokens, skip_special_tokens = True)
                greedy_text = self.tokenizer.decode(greedy_tokens, skip_special_tokens = True)
                max_text_equal = (greedy_text == target_text)

                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal or max_text_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

 

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
    if "lambada_openai" not in task_names and "lambada_standard" not in task_names:
        print(f"Only work for lambada test")
        return

    task_dict = tasks.get_task_dict(task_names)

    model = T5ModelLambada(args.model,device=args.device, batch_size=args.batch_size)
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
