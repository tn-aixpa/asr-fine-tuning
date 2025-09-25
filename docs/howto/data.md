
## Data Overview and Preparaton Process

The expected input data should have a form of audio datasets structured following the Huggingface practices for Audio datasets.
See for example [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0). 

The preprocessing procedure downloads the whole datasets, preprocesses and samples the corresponding data slices, and saves the 
data in a way it can be used in training. The resulting processed data is then uploaded to the platform as an artifact. 

**Note:** If you want to use your own dataset, you should first upload it to Hugging Face Hub.  

## 📝 Dataset Format

Dataset must be organized as the following structre
```
audio
  lang_code1
     train
       lang_code1_train_0.tar 
       ...
     test
     validate
     ...
  lang_code2
  ...
transcript
  lang_code1
     train
     test
     validate
     ...
  lang_code2
  ...
```

That is, it should contain audio and transcripts, with a dedicated folder for each subset (language code), with subfolders for specific data parts - train, validate, etc.

The training procedure relies on the ``train``, ``validate`` and ``test`` parts present.

## 🧹 Dataset Preparation

Use the following procedure for the data preprocessing:

1. Define the processing function

```python

import  digitalhub as dh

project = dh.get_or_create_project("audio-training")

func = project.new_function(
    name="create-dataset", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="src/fine_tuning_seq2seq.py",  
    handler="preprocess_dataset",
    requirements=["datasets[audio]==3.6.0", "transformers==4.56.1", "torch==2.8.0", "accelerate==1.10.1", "evaluate==0.4.5", "jiwer==4.0.0"]
)
```

2. Run the function as job.

```python
train_run = func.run(action="job",
                     parameters={
                         "model_id": "openai/whisper-small",
                         "artifact_name": "audio-dataset",
                         "dataset_name": "mozilla-foundation/common_voice_17_0",
                         "language": "Italian",
                         "language_code": "it",
                         "max_train_samples": 100,
                         "max_eval_samples": 100
                     },
                     secrets=["HF_TOKEN"],
                     envs=[
                        {"name": "HF_HOME", "value": "shared/data/huggingface"},
                        {"name": "TRANSFORMERS_CACHE", "value":  "shared/data/huggingface"}
                     ],
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "volume-llmpa",
                        "mount_path": "/shared/data",
                        "spec": { "size": "300Gi" }}]
					)
```

Here 

- ``model_id`` refers to the base model to fine-tune, 
- ``artifact_name`` defines the name of the artifact to store,
- ``dataset_name`` defines the name of the dataset on the Huggingface,
- ``language`` the language to use for training and ``language_code`` the subset of data to consider
- ``max_train_samples`` and ``max_eval_samples`` define a number of samples to consider (for debug purpose only).

The environment variables are used to map the Huggingface cache folder to the mounted path of the volume in case of large datasets should be processed.