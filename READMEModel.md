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
