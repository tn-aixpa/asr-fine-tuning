# Train LLM Adapter

It is possible to run the training as a python job on the platform. The following steps may be performed directly from the
jupyter notebook workspace on the platform. Clone the repo within the workspace and perform the following steps.

1. Define the training function

```python

import  digitalhub as dh

project = dh.get_or_create_project("audio-training")

func = project.new_function(
    name="train-audio-model", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="src/fine_tuning_seq2seq.py",  
    handler="train_and_log_model",
    requirements=["datasets[audio]==3.6.0", "transformers==4.52.0", "torch==2.8.0", "accelerate==1.10.1", "evaluate==0.4.5", "jiwer==4.0.0"]
)
```

2. Run the function as job.

It is important to note that for the execution a volume should be created as the space requirements exceed the default available space.

The huggingface token should be passed as project secrets or as env variables (not recommended). Create the corresponding secret (``HF_TOKEN``) in the project configuration.

```python
train_run = func.run(action="job",
                     parameters={
                         "model_id": "openai/whisper-small",
                         "model_name": "whisper-ft",
                         "dataset_name": "mozilla-foundation/common_voice_11_0",
                         "language": "Italian",
                         "language_code": "it",
                         "max_train_samples": 100,
                         "max_eval_samples": 100,
                         "eval_steps": 100,
                         "save_steps": 100,
                         "max_steps": 500,
                         "warmup_steps": 50
                     },
                     profile="1xa100",
                     secrets=["HF_TOKEN"],
                     envs=[
                        {"name": "HF_HOME", "value": "shared/data/huggingface"},
                        {"name": "TRANSFORMERS_CACHE", "value":  "shared/data/huggingface"}
                     ],
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "volume-llmpa",
                        "mount_path": "/shared/data",
                        "spec": { "size": "100Gi" }}]
					)
```

Here 

- ``model_id`` refers to the base model to fine-tune, 
- ``model_name`` define the name of the target model entity,
- ``dataset_name`` defines the name of the dataset on the Huggingface,
- ``language`` the language to use for training and ``language_code`` the subset of data to consider
- ``max_train_samples`` and ``max_eval_samples`` define a number of samples to consider (for debug purpose only).

The environment variables are used to map the Huggingface cache folder to the mounted path of the volume in case of large datasets should be processed.