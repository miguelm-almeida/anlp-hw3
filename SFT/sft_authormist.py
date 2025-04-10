import os
import torch
import pandas as pd
import csv
from datasets import Dataset
from tqdm import tqdm
import bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # === STEP 1: Load CSV and prepare Dataset ===
    df = pd.read_csv("subset_data.csv")
    df = df[df['model'].isin(['llama-3', 'gpt-4o'])].reset_index(drop=True)

    hf_dataset = Dataset.from_pandas(df[['response']].rename(columns={"response": "text"}))

    # === STEP 2: Load model in 8-bit mode + LoRA config ===
    model_name = "authormist/authormist-originality"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        load_in_8bit_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === STEP 3: Attach LoRA adapters ===
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # adjust if needed
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # === STEP 4: Tokenize dataset ===
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = hf_dataset.map(tokenize, batched=True)

    # === STEP 5: SFT Training Config ===
    training_args = SFTConfig(
        output_dir="./authormist_sft_lora",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_total_limit=2,
        save_steps=500,
        logging_steps=50,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
    )

    trainer.train()

    # === STEP 6: Generate and Save to CSV (safe quoting) ===
    model.eval()

    output_csv = "paraphrased_results.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Original Text", "Generated Text"])

        for text in tqdm(df["response"].tolist(), desc="Generating from fine-tuned model"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            writer.writerow([text, generated])

    print(f"\nInference complete. Output saved to: {output_csv}")

if __name__ == "__main__":
    main()