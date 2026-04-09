#!/usr/bin/env python3
"""
Read a parquet file with LLM responses, compute token counts per response,
and print summary statistics (mean, std, min, max, percentiles).
"""
import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def _is_present(val) -> bool:
    """True if value is present (not NA). Safe for scalars and arrays."""
    if isinstance(val, np.ndarray):
        return bool(np.any(pd.notna(val)))
    return bool(pd.notna(val))


def _cell_to_strings(val) -> list[str]:
    """Turn one cell (row) value into a flat list of response strings.
    Each row may be a single string, or an array/list of N responses (e.g. 16).
    All are flattened so we get one string per response.
    """
    if isinstance(val, (list, np.ndarray)):
        strings = []
        for r in (val if isinstance(val, list) else val.tolist()):
            if isinstance(r, (list, np.ndarray)):
                strings.extend(_cell_to_strings(r))
            else:
                strings.append(str(r) if _is_present(r) else "")
        return strings
    return [str(val) if _is_present(val) else ""]


def get_response_strings(df: pd.DataFrame, response_key: str = "response") -> list[str]:
    """Collect one response string per row (the first when the cell has multiple)."""
    if response_key not in df.columns:
        if "responses" in df.columns:
            response_key = "responses"
        else:
            raise KeyError(
                f"Neither 'response' nor 'responses' found. Columns: {list(df.columns)}"
            )

    out = []
    for val in df[response_key]:
        strings = _cell_to_strings(val)
        out.append(strings[0] if strings else "")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute token-length statistics for the 'response' field in a parquet file."
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to the parquet file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/llama-3.1-8b",
        help="HuggingFace tokenizer name for counting tokens (default: llama-3.1-8b). "
        "Use the same tokenizer as your LLM for accurate counts.",
    )
    parser.add_argument(
        "--response-key",
        type=str,
        default="response",
        help="Column name for response text (default: response). "
        "If not present, 'responses' is tried.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Column name for prompt text to display alongside shortest responses (default: prompt). "
        "If not present, prompts are omitted.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-row response counts and content to debug multiple responses per row.",
    )
    parser.add_argument(
        "--debug-max-rows",
        type=int,
        default=10,
        help="When --debug, only print full content for this many rows (default: 10).",
    )
    parser.add_argument(
        "--num-shortest",
        type=int,
        default=0,
        help="If >0, print this many globally shortest responses by token length (default: 0, disabled).",
    )
    parser.add_argument(
        "--num-middle",
        type=int,
        default=0,
        help="If >0, print this many responses near the median token length (default: 0, disabled).",
    )
    parser.add_argument(
        "--num-longest",
        type=int,
        default=0,
        help="If >0, print this many globally longest responses by token length (default: 0, disabled).",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=None,
        metavar="P",
        help="Filter to rows with token length >= P-th percentile (e.g. 25). Requires --max-percentile and --output.",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=None,
        metavar="P",
        help="Filter to rows with token length <= P-th percentile (e.g. 75). Requires --min-percentile and --output.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path when using --min-percentile and --max-percentile.",
    )
    parser.add_argument(
        "--filter-func-matches",
        action="store_true",
        help="If set, keep only rows where every func_i expression (i numeric) in the prompt appears in the response text.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    if args.response_key in df.columns:
        response_key = args.response_key
    elif "response" in df.columns:
        response_key = "response"
    elif "responses" in df.columns:
        response_key = "responses"
    else:
        raise KeyError(
            f"No response column found. Columns: {list(df.columns)}"
        )

    # Determine prompt column (optional; used for func_i checks and displaying shortest examples)
    if args.prompt_key in df.columns:
        prompt_key = args.prompt_key
    else:
        prompt_key = None

    # Count how many prompts contain 0, 1, or 2+ func_i expressions (in original file)
    if prompt_key is not None:
        func_pattern = re.compile(r"func_\d+")
        count_0 = count_1 = count_2_or_more = 0
        for _, row in df.iterrows():
            prompt_val = row[prompt_key]
            prompt_text = str(prompt_val) if _is_present(prompt_val) else ""
            num_funcs = len(func_pattern.findall(prompt_text))
            if num_funcs == 0:
                count_0 += 1
            elif num_funcs == 1:
                count_1 += 1
            else:
                count_2_or_more += 1
        print("func_i occurrences per prompt (original data):")
        print(f"  0 occurrences: {count_0}")
        print(f"  1 occurrence : {count_1}")
        print(f"  2+ occurrences: {count_2_or_more}")
        print()

    # Debug: how many responses per row and what they look like
    if args.debug:
        col = df[response_key]
        n_rows = len(df)
        print("=== DEBUG: response column structure ===")
        print(f"Column name: '{response_key}'")
        print(f"Total rows: {n_rows}")
        responses_per_row = []
        for i, val in enumerate(col):
            if isinstance(val, list):
                n = len(val)
                responses_per_row.append(n)
            elif isinstance(val, np.ndarray):
                responses_per_row.append(int(np.sum(pd.notna(val))))
            else:
                responses_per_row.append(1 if pd.notna(val) else 0)
        responses_per_row = np.array(responses_per_row)
        print(f"Responses per row: min={responses_per_row.min()}, max={responses_per_row.max()}, mean={responses_per_row.mean():.2f}")
        dist = Counter(responses_per_row)
        print(f"Distribution (count of rows with N responses): {dict(sorted(dist.items()))}")
        print()
        max_show = min(n_rows, args.debug_max_rows)
        max_preview = 200
        for i in range(max_show):
            val = col.iloc[i]
            if isinstance(val, (list, np.ndarray)):
                vals = list(val) if isinstance(val, np.ndarray) else val
                print(f"--- Row {i} ({len(vals)} response(s)) ---")
                for j, r in enumerate(vals):
                    s = str(r) if _is_present(r) else "<NA>"
                    preview = s[:max_preview] + ("..." if len(s) > max_preview else "")
                    print(f"  [{j}] len={len(s)}: {repr(preview)}")
            else:
                s = str(val) if _is_present(val) else "<NA>"
                preview = s[:max_preview] + ("..." if len(s) > max_preview else "")
                print(f"--- Row {i} (1 response) ---")
                print(f"  [0] len={len(s)}: {repr(preview)}")
        if n_rows > max_show:
            print(f"... ({n_rows - max_show} more rows not shown; use --debug-max-rows to see more)")
        print("=== END DEBUG ===\n")

    texts = get_response_strings(df, response_key)
    if not texts:
        print("No response strings found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = []
    for t in tqdm(texts):
        enc = tokenizer(t, return_tensors=None, add_special_tokens=False)
        lengths.append(len(enc["input_ids"]))

    lengths = np.array(lengths)
    n = len(lengths)

    print(f"Parquet: {args.parquet_path}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Number of responses (one per row, first only): {n}")
    print()
    print("Token length statistics:")
    print(f"  mean   = {lengths.mean():.2f}")
    print(f"  std    = {lengths.std():.2f}")
    print(f"  min    = {lengths.min()}")
    print(f"  max    = {lengths.max()}")
    print(f"  median = {np.median(lengths):.0f}")
    print(f"  25%    = {np.percentile(lengths, 25):.0f}")
    print(f"  75%    = {np.percentile(lengths, 75):.0f}")
    print(f"  90%    = {np.percentile(lengths, 90):.0f}")
    print(f"  95%    = {np.percentile(lengths, 95):.0f}")
    print(f"  99%    = {np.percentile(lengths, 99):.0f}")

    # Optionally show some example responses at different lengths
    show_shortest = bool(args.num_shortest and args.num_shortest > 0)
    show_middle = bool(args.num_middle and args.num_middle > 0)
    show_longest = bool(args.num_longest and args.num_longest > 0)

    def _print_examples(label: str, indices: np.ndarray) -> None:
        if len(indices) == 0:
            return
        print()
        print(f"{len(indices)} {label} responses by token length:")
        for rank, idx in enumerate(indices, start=1):
            length = lengths[idx]
            text = texts[idx]
            print(f"--- {label.capitalize()} #{rank} (row {idx}, {length} tokens) ---")
            if prompt_key is not None:
                prompt_val = df.iloc[idx][prompt_key]
                prompt_text = str(prompt_val) if _is_present(prompt_val) else "<NA>"
                print("PROMPT:")
                print(prompt_text)
                print()
            print("RESPONSE:")
            print(text)
            print()

    if show_shortest or show_middle or show_longest:
        sorted_indices = np.argsort(lengths)

        if show_shortest:
            k = min(args.num_shortest, n)
            shortest_indices = sorted_indices[:k]
            _print_examples("shortest", shortest_indices)

        if show_middle:
            k = min(args.num_middle, n)
            start = max(0, (n - k) // 2)
            middle_indices = sorted_indices[start : start + k]
            _print_examples("middle", middle_indices)

        if show_longest:
            k = min(args.num_longest, n)
            longest_indices = sorted_indices[-k:][::-1]
            _print_examples("longest", longest_indices)

    # Optional: filter rows and write
    use_percentile_filter = args.min_percentile is not None or args.max_percentile is not None
    use_func_filter = args.filter_func_matches

    if use_percentile_filter or use_func_filter:
        if args.output is None:
            raise ValueError("--output must be provided when using any filtering option.")

        mask = np.ones(n, dtype=bool)

        # Filter by func_i matching between prompt and response
        if use_func_filter:
            if prompt_key is None:
                raise ValueError(
                    "--filter-func-matches requested but prompt column not found; "
                    "set --prompt-key to an existing column."
                )
            func_pattern = re.compile(r"func_\d+")
            func_mask = np.ones(n, dtype=bool)
            for idx in range(n):
                prompt_val = df.iloc[idx][prompt_key]
                prompt_text = str(prompt_val) if _is_present(prompt_val) else ""
                funcs = func_pattern.findall(prompt_text)
                if not funcs:
                    continue  # no constraints from this row
                resp_text = texts[idx]
                for f in funcs:
                    if f not in resp_text:
                        func_mask[idx] = False
                        break
            mask &= func_mask

        # Filter by percentile range
        if use_percentile_filter:
            if args.min_percentile is None or args.max_percentile is None:
                raise ValueError("--min-percentile and --max-percentile must both be set when using percentile filtering.")
            if not (0 <= args.min_percentile <= 100 and 0 <= args.max_percentile <= 100):
                raise ValueError("Percentiles must be between 0 and 100.")
            if args.min_percentile > args.max_percentile:
                raise ValueError("--min-percentile must be <= --max-percentile.")
            threshold_lo = np.percentile(lengths, args.min_percentile)
            threshold_hi = np.percentile(lengths, args.max_percentile)
            percentile_mask = (lengths >= threshold_lo) & (lengths <= threshold_hi)
            mask &= percentile_mask

        n_before = len(df)
        df_filtered = df[mask].copy()
        n_after = len(df_filtered)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_filtered.to_parquet(args.output, index=False)
        print()
        if use_func_filter:
            print("Applied func_i prompt/response consistency filter.")
        if use_percentile_filter:
            print(
                f"Filter: token length in [{args.min_percentile}th, {args.max_percentile}th] percentile "
                f"-> [{threshold_lo:.0f}, {threshold_hi:.0f}] tokens"
            )
        print(f"Rows: {n_before} -> {n_after} (removed {n_before - n_after})")
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
