import pandas as pd
import torch
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Login using your Hugging Face token (***-marked for privacy)
login("hf_***", add_to_git_credential=True)


models = [

    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'Qwen/Qwen2.5-0.5B-Instruct',
    'distilgpt2',
    'StabilityAI/stablelm-tuned-alpha-3b',
    'gpt2'
]

TOKENIZER_MAX_SEQ_LEN = 512  # Maximum sequence length for the model
BATCH_SIZE = 2
MAX_NEW_TOKENS = 50
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Adjust device to use CUDA or CPU

# Load and clean dataset
df = pd.read_excel("InitialData.xlsx")  # Make sure to adjust the file path if needed
df = df[['xi']].dropna()
df.to_csv('cleaned_data.csv', index=False)
dataset = load_dataset("csv", data_files="cleaned_data.csv", split="train")

def load_model(model_name):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model, device mapping to auto, which will decide based on device availability
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        device_map='auto' if torch.cuda.is_available() else 'cpu',
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_completions_for_model(model_name, dataset):
    all_generated_answers = []
    model, tokenizer = load_model(model_name)

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i:i + BATCH_SIZE]
        modified_batch = ["Complete the sentence: " + xi for xi in batch['xi']]

        inputs = tokenizer(modified_batch, padding=True, truncation=True, max_length=TOKENIZER_MAX_SEQ_LEN, return_tensors="pt")
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        xj_completions = [text[len(prompt):].strip() for text, prompt in zip(generated_texts, modified_batch)]

        # Store the xi, xj, full sentence, and model used
        for xi, xj in zip(batch['xi'], xj_completions):
            all_generated_answers.append({
                "xi": xi,
                "xj": xj,
                "full_sentence": xi + " " + xj,
                "model_used": model_name.split('/')[-1] if '/' in model_name else model_name
            })

    return all_generated_answers

# Run each model sequentially and consolidate results
all_results = []

for model_name in models:
    try:
        model_results = generate_completions_for_model(model_name, dataset)
        all_results.extend(model_results)
    except Exception as exc:
        print(f"Error with model {model_name}: {exc}")

# Save the results to a CSV file
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_df = pd.DataFrame(all_results)
output_df.to_csv(os.path.join(results_dir, 'Dataset.csv'), index=False)

print("Results saved.")
