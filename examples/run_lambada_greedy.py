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

    task_dict = tasks.get_task_dict(task_names)

    model = models.huggingface.AutoCausalLM(args.model,device=args.device, batch_size=args.batch_size)
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
