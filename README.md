
# Curating the Dataset: Sentence Completion with Language Models

This project utilizes various language models to generate completions for sentences. It processes a dataset of initial sentences (referred to as `xi`), prompts the models to complete them, and saves the results in a CSV file.

## Prerequisites

To run this project, you need to have Python installed and a Hugging Face account to access the models.

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install required packages**:
   Use the following command to install the necessary Python libraries:
   ```bash
   pip install pandas torch transformers datasets huggingface_hub tqdm openpyxl
   ```

4. **Log into Hugging Face**:
   Run the following command and follow the prompts:
   ```bash
   huggingface-cli login
   ```
   Enter your Hugging Face token when prompted (found in your Hugging Face account settings).

## Usage

1. **Prepare your dataset**:
   Create an Excel file named `InitialData.xlsx` with a sheet containing a column labeled `xi` that includes the initial sentences you want to complete.

2. **Run the script**:
   Execute the Python script that processes the sentences. This script loads the `InitialData.xlsx`, cleans the dataset by keeping only the `xi` column, and saves it as a CSV file named `cleaned_data.csv`. It then loads various language models and generates completions for each sentence in the dataset.

3. **Check the results**:
   After the script finishes running, it will create a folder named `results` and save a CSV file named `Dataset.csv` inside it. This file contains the initial sentences, the generated completions, and the names of the models used.

## Code Explanation

The code starts by importing necessary libraries, including Pandas for data manipulation and Transformers for working with language models. It logs into Hugging Face to access various pre-trained models. The models are defined in a list, and the script sets parameters for maximum sequence length, batch size, and the number of new tokens to generate.

The dataset is loaded from an Excel file, cleaned to retain only the relevant column, and saved as a CSV. Each model is loaded in a function that initializes the tokenizer and model. The script processes the input sentences in batches, prepends a prompt, and generates completions using the models. The results, including the original and completed sentences, are stored in a list and saved as a CSV file.

The project utilizes several pre-trained language models from the Hugging Face Model Hub to generate text completions. The models selected for this project include **TinyLlama/TinyLlama-1.1B-Chat-v1.0**, **Qwen/Qwen2.5-0.5B-Instruct**, **distilgpt2**, **StabilityAI/stablelm-tuned-alpha-3b**, and **gpt2**. These models are well-regarded for their capabilities in natural language processing tasks. Each model is initialized using a function that loads the corresponding tokenizer and model, which adapts to the available hardware (either GPU or CPU) for efficient computation.

The script processes the dataset in batches, where each batch is padded and truncated to a maximum sequence length of 512 tokens. For each input sentence (referred to as `xi`), the models generate a completion, which is a continuation or response to the input sentence. The generated completions are refined through parameters such as `top_k` and `top_p`, which control the sampling strategy to produce diverse outputs while avoiding incoherent text. The results are collected, including the original sentences, generated completions (denoted as `xj`), the full sentence combining both, and the name of the model used for generation. This structured approach enables the effective exploration of various language models and their performance on the sentence completion task.


# Running the Superior Model
## Prerequisites

Make sure you have the following installed:
- Python 3.7 or higher
- PyTorch
- Transformers
- Sentence Transformers
- scikit-learn
- Matplotlib
- pandas
- huggingface_hub

You can install the required packages using pip.

## Setup

Ensure you have access to the Hugging Face Hub to download the models. You can log in using your Hugging Face token.

## Dataset

Prepare your dataset in CSV format. The dataset should have the following columns:
- `xi`: Input prompts (if applicable)
- `xj`: Truncated output prompts
- `full_sentence`: Complete sentences (if applicable)
- `model_used`: Labels indicating which model generated the completion

Place your dataset file in the appropriate directory. The instructions to dataset is also in the Dataset Folder.

## Code Structure

The project includes the following key components:
- **Imports**: Necessary libraries are imported, including PyTorch, Hugging Face Transformers, pandas, and scikit-learn.
- **Data Loading**: The dataset is loaded from a CSV file, and missing values are filled.
- **Label Encoding**: The labels are encoded using `LabelEncoder` from scikit-learn.
- **Custom Dataset Class**: The `TextDataset` class prepares the data for training and testing.
- **Model Training and Evaluation**: The function `train_and_evaluate` handles the training and evaluation of the model, using various hyperparameters.
- **Hyperparameter Experimentation**: A grid search over batch sizes, learning rates, optimizers, and schedulers is performed to find the best configuration.

## Training and Evaluation

The `train_and_evaluate` function is designed to train the RoBERTa model and evaluate its performance. It performs the following steps:
1. **Data Loading**: Loads training and testing data using PyTorch's `DataLoader`.
2. **Model Initialization**: Initializes the RoBERTa model for sequence classification.
3. **Training Loop**: Trains the model for a specified number of epochs, calculating the loss at each epoch.
4. **Evaluation**: Evaluates the model on the test dataset and generates a classification report, including precision, recall, and F1 score.
5. **Loss Plotting**: Plots the training loss over epochs for visual analysis.

## Running Experiments

To run the training and evaluation experiments, execute the main code block that iterates through all combinations of hyperparameters. Adjust the parameters as needed.

### Hyperparameters to Experiment With:
- Batch Sizes: [16, 32]
- Learning Rates: [1e-5, 2e-5, 5e-5]
- Optimizers: ['AdamW', 'Adam', 'Adamax']
- Schedulers: ['cosine', 'linear']

## Results

After running the experiments, the classification report will be printed to the console for each hyperparameter configuration. Additionally, loss plots will be generated to visualize the training process.
