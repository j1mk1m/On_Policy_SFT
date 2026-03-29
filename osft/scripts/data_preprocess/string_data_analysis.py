import re
import json
import argparse
from collections import Counter

import pandas as pd

func_name_mapping = {
    "func_0": "deterministic_shuffle",
    "func_1": "repeat_str",
    "func_2": "remove_vowels",
    "func_3": "sort_chars",
    "func_4": "reverse_words",
    "func_5": "add_prefix",
    "func_6": "add_suffix",
    "func_7": "interlace_str",
    "func_8": "rotate_str",
    "func_9": "mirror_str",
    "func_10": "alternate_case",
    "func_11": "shift_chars",
    "func_12": "vowel_to_number",
    "func_13": "insert_separator",
    "func_14": "duplicate_every_char",
    "func_15": "fancy_brackets",
    "func_16": "compress_repeats",
    "func_17": "recursive_reverse",
    "func_18": "loop_concat",
    "func_19": "while_rotate",
    "func_20": "recursive_interlace",
    "func_21": "loop_filter_nonalpha",
    "func_22": "verify_even_length",
    "func_23": "backchain_add_digit",
    "func_24": "backchain_palindrome",
}

# Matches func_i only when followed by '(' (i.e. a call site, not a definition)
CALL_PATTERN = re.compile(r'\b(func_\d+)\s*\(')


def extract_combo(code: str) -> tuple:
    """
    Extract the ordered composition chain of func_i calls from the return
    expression of main_solution, outermost first.

    e.g. 'return func_9(func_3(x))' → ('func_9', 'func_3')
         'return func_1(x, 2)'       → ('func_1',)
    """
    # Isolate the return expression (last line of main_solution)
    return_line = code.split('return ')[-1]
    # Read calls in left-to-right order = outermost to innermost
    return tuple(CALL_PATTERN.findall(return_line))


def func_id_to_name(func_id: str) -> str:
    return func_name_mapping.get(func_id, func_id)


def combo_label(combo: tuple) -> str:
    """Human-readable label: 'mirror_str → sort_chars'"""
    return ' → '.join(func_id_to_name(f) for f in combo)


def analyze(parquet_path: str) -> None:
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from '{parquet_path}'")

    combo_counts: Counter = Counter()

    for _, row in df.iterrows():
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        code = ground_truth["ref_code"]
        combo = extract_combo(code)
        combo_counts[combo] += 1

    # Determine the depth of problems in this dataset
    depths = sorted({len(c) for c in combo_counts})
    print(f"Depths found: {depths}")

    for depth in depths:
        depth_combos = {c: n for c, n in combo_counts.items() if len(c) == depth}
        total = sum(depth_combos.values())
        print(f"\n--- Depth {depth} ({total} samples, {len(depth_combos)} unique combinations) ---")

        if depth == 1:
            # Single-function view: sort by func index
            print(f"\n{'func_id':<12} {'name':<28} {'count':>8}")
            print("-" * 52)
            for (func_id,), count in sorted(depth_combos.items(), key=lambda x: int(x[0][0].split('_')[1])):
                print(f"{func_id:<12} {func_id_to_name(func_id):<28} {count:>8}")
            print("-" * 52)
            print(f"{'Total':<41} {total:>8}")
        else:
            # Multi-function view: sort by combo label alphabetically
            col_width = max(len(combo_label(c)) for c in depth_combos) + 2
            print(f"\n{'combination':<{col_width}} {'count':>8}")
            print("-" * (col_width + 10))
            for combo, count in sorted(depth_combos.items(), key=lambda x: combo_label(x[0])):
                print(f"{combo_label(combo):<{col_width}} {count:>8}")
            print("-" * (col_width + 10))
            print(f"{'Total':<{col_width}} {total:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze function combination usage in a string dataset parquet file.")
    parser.add_argument("parquet_path", help="Path to the parquet file to analyze.")
    args = parser.parse_args()

    analyze(args.parquet_path)
