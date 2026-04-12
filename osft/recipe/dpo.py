"""
Minimal standalone TRL DPO example.

Run (single GPU):
    python recipe/dpo.py
Run (multi GPU):
    torchrun --standalone --nproc_per_node=4 recipe/dpo.py
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def main() -> None:
    train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    training_args = DPOConfig(
        output_dir="checkpoints/dpo_example",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=1,
        bf16=True,
        beta=0.1,
        max_length=2048,
        max_prompt_length=1024,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=MODEL_NAME,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()