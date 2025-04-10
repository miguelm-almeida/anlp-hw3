import time
import torch
import csv
import gc
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1:.2f} seconds")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Invalid lexical diversity."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Invalid order diversity."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

            del final_input, outputs
            gc.collect()
            torch.cuda.empty_cache()

        return output_text.strip()

def main():
    # === CONFIGURATION ===
    df = pd.read_csv('subset_data.csv')
    df = df[df['model'].isin(['llama-3', 'gpt-4o'])]
    ai_texts = df['response'].tolist()
    output_csv = "paraphrased_results.csv"
    prefix_prompt = "Please paraphrase the following text to make it more human-like while preserving the original meaning."

    # Initialize paraphraser
    dp = DipperParaphraser()

    # Paraphrase and write results
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Text", "Paraphrased Text"])

        for ai_text in tqdm(ai_texts, desc="Paraphrasing with DIPPER"):
            try:
                paraphrased = dp.paraphrase(
                    input_text=ai_text,
                    lex_diversity=60,
                    order_diversity=20,
                    prefix=prefix_prompt,
                    do_sample=True,
                    top_p=0.75,
                    max_length=512
                )
                print(paraphrased)
                writer.writerow([ai_text, paraphrased])
            except Exception as e:
                print(f"Failed to paraphrase text: {ai_text[:50]}... due to {e}")
                writer.writerow([ai_text, "ERROR"])

    print(f"\nParaphrasing complete. Output saved to {output_csv}")

if __name__ == "__main__":
    main()
