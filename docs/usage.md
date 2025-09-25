# Automated Speech Recogniction Fine Tuner

Automated Speech Recogniction Fine Tuner represents a transformers-based implementation of the fine-tuning algorithm aiming to produce a dataset a transcribe or translate models like, e.g., Whisper.

## 🔧 Prerequisites

- Python: Version 3.10 or higher.

- GPU: Required for training.
  
- API Tokens:

    - Hugging Face token with access to Llama-3.1-8B-Instruct.Create one [here](https://huggingface.co/settings/tokens). If you want to use public model, you don't need this parameter.

    - Weights & Biases API key for logging (optional).Create one [here](https://wandb.ai/home). This parameter is optional.

## Hardware Requirements

The training procedure requires a certain model of space (depends on the base model, dataset, and number of checkpoints) and GPU. 

## Train Dataset

To perform the training, the dataset should be structured in a manner compatible with Transformer Datasets structure and is
expected to be published there. To facilitate the training in case only a subset of data is needed, it is possible to 
use a dedicated procedure that download, prepares datasets, and prepares the data that may be used directly (should be stored
to the cache directory of Huggingface library). The following preprocessing parameters are accepted:

| Argument                | Type    | Description                                                                                 |
|-------------------------|---------|---------------------------------------------------------------------------------------------|
| artifact_name           | str     | Nome dell'artefatto da quale leggere il dataset preprocessato per il dataset di riferimento | 
| model_id                | str     | Model ID (e.g., huggingface repo/model name)                                                |
| dataset_name            | str     | Name of the dataset on Hugging Face Hub                                                     |
| language_code           | str     | Name of the data subset (e.g., in the form of two-letter ISO language code). Default  'it'  |
| language                | str     | Language to use (defaults to Italian)                                                       |
| max_train_samples       | int     | Maximum number of rows of training dataset to consider. For debug purpose.                  |
| max_eval_samples        | int     | Maximum number of rows of evaluation dataset to consider. For debug purpose.                |
|-------------------------|---------|---------------------------------------------------------------------------------------------|


## Training parameters

The training procedure accepts the following parameters (all must be provided unless marked as optional):

| Argument                | Type    | Description                                                                                 |
|-------------------------|---------|---------------------------------------------------------------------------------------------|
| model_name              | str     | Model name with which publish the model                                                     |
| model_id                | str     | Model ID (e.g., huggingface repo/model name)                                                |
| dataset_name            | str     | Name of the dataset on Hugging Face Hub                                                     |
| artifact_name           | str     | Nome dell'artefatto da quale leggere il dataset preprocessato per il dataset di riferimento | 
| language_code           | str     | Name of the data subset (e.g., in the form of two-letter ISO language code). Default  'it'  |
| language                | str     | Language to use (defaults to Italian)                                                       |
| max_sequence_length     | int     |  Maximum sequence length (255)                                                              |
| learning_rate           | float   | Learning rate (1e-5)                                                                        |
| train_batch_size        | int     | Training batch size (16)                                                                    |
| eval_batch_size         | int     | Evaluation batch size (8)                                                                   |
| grad_accum_steps        | int     | Gradient accumulation steps (1)                                                             |
| logging_steps           | int     | Number of steps between logging (25)                                                        |
| eval_steps              | int     | Number of steps between evaluations (1000)                                                  |
| save_steps              | int     | Number of steps between model checkpoints (1000)                                            |
| warmup_steps            | int     | Warmup steps to use before actual training (500)                                            |
| max_steps               | int     | Maximum number of steps (5000)                                                              |
| max_train_samples       | int     | Maximum number of rows of training dataset to consider. For debug purpose.                  |
| max_eval_samples        | int     | Maximum number of rows of evaluation dataset to consider. For debug purpose.                |
|-------------------------|---------|---------------------------------------------------------------------------------------------|


  ## Execution modalities

  The traning may be performed as follows:
  
  - as a [python job in the platform](./howto/train_container.md)

  The preprocessing may be performed as follows:
  
  - as a [python job in the platform](./howto/train_container.md)

- **Preprocessing Outputs**

Preprocessed data is stored as a folder compatible with the dataset cache of the Transformer library.

- **Training Outputs**

Model data with configurations, specs, ecc and the last checkpoint considered.