from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm
import os
import random

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from verl.utils.reward_score.codeio import compute_score


parser = ArgumentParser()
parser.add_argument(
    "--gen_path",
    type=str,
    default="gen_path.parquet",
)
parser.add_argument("--save_path", type=str, default="data/string_manipulation_dpo_difficult")
parser.add_argument("--val_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--no_remove_context", action="store_true")
parser.add_argument("--remove_overlong", action="store_true")
parser.add_argument(
    "--pair_mode",
    type=str,
    default="all",
    choices=["all", "one_to_one"],
    help="all: cartesian product of positives/negatives, one_to_one: random pairing up to min(pos, neg)",
)
parser.add_argument(
    "--max_pairs_per_prompt",
    type=int,
    default=-1,
    help="Cap number of (chosen, rejected) pairs for each prompt; -1 means no cap.",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

if args.max_length:
    assert args.tokenizer is not None, "--tokenizer must be set when --max_length is used"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
else:
    tokenizer = None


def _valid_response(response: str) -> bool:
    if args.remove_overlong and not response.strip().endswith("<|eot_id|>"):
        return False
    if tokenizer is not None and len(tokenizer(response)["input_ids"]) > args.max_length:
        return False
    return True


def _build_prompt(raw_prompt: str) -> str:
    if args.no_remove_context:
        return raw_prompt
    start_idx = raw_prompt.find("def main_solution(x):")
    if start_idx == -1:
        return f"You are given a code:\n\n{raw_prompt}"
    return f"You are given a code:\n\n{raw_prompt[start_idx:]}"


def _pair_samples(positives, negatives):
    if args.pair_mode == "one_to_one":
        pos = positives[:]
        neg = negatives[:]
        random.shuffle(pos)
        random.shuffle(neg)
        n_pairs = min(len(pos), len(neg))
        pairs = [(pos[i], neg[i]) for i in range(n_pairs)]
    else:
        pairs = [(p, n) for p in positives for n in negatives]

    if args.max_pairs_per_prompt > 0 and len(pairs) > args.max_pairs_per_prompt:
        pairs = random.sample(pairs, args.max_pairs_per_prompt)
    return pairs


all_gen_path = []
if os.path.isdir(args.gen_path):
    for file in os.listdir(args.gen_path):
        if file.endswith(".parquet"):
            all_gen_path.append(os.path.join(args.gen_path, file))
    all_gen_path.sort()
else:
    all_gen_path = [args.gen_path]

new_dataset = []
func_statistics = defaultdict(lambda: 0)
pair_statistics = Counter()

for gen_path in all_gen_path:
    gen_dataset = load_dataset("parquet", data_files=gen_path)["train"]

    for data in tqdm(gen_dataset):
        positives = []
        negatives = []
        for response in data["responses"]:
            if not _valid_response(response):
                continue
            is_correct = compute_score(
                response, data["reward_model"]["ground_truth"], data["data_source"]
            )[1]
            if is_correct == 1:
                positives.append(response)
            else:
                negatives.append(response)

        if not positives or not negatives:
            continue

        prompt = _build_prompt(data["prompt"][0]["content"])
        pairs = _pair_samples(positives, negatives)
        pair_statistics["total_pairs"] += len(pairs)
        pair_statistics["prompts_with_pairs"] += 1

        for chosen, rejected in pairs:
            new_data = {
                "data_source": data["data_source"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "ability": data["ability"],
                "reward_model": {
                    "style": data["reward_model"]["style"],
                    "ground_truth": data["reward_model"]["ground_truth"],
                },
                "extra_info": {
                    "split": data["extra_info"]["split"],
                    "index": data["extra_info"]["index"],
                },
            }
            func_statistics[prompt.split("return ")[-1].split("(")[0]] += 1
            new_dataset.append(new_data)

if len(new_dataset) == 0:
    raise ValueError("No valid DPO pairs were created. Please relax filtering or check input files.")

print("Pair statistics:", dict(pair_statistics))
print("Function statistics:", dict(func_statistics))

new_dataset = Dataset.from_list(new_dataset)
print("Data sources:", Counter(new_dataset["data_source"]).items())

os.makedirs(args.save_path, exist_ok=True)
if args.val_size:
    split_dataset = new_dataset.train_test_split(test_size=args.val_size, seed=args.seed)
    train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    train_dataset.to_parquet(f"{args.save_path}/train.parquet")
    test_dataset.to_parquet(f"{args.save_path}/test.parquet")
else:
    new_dataset.to_parquet(f"{args.save_path}/train.parquet")
    print(f"Saved {len(new_dataset)} data.")
