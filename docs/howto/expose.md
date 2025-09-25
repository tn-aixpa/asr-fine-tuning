# Expose model as a service using KubeAI

KubeAI supports exposing the FasterWhisper models as automated speech recognition services following OpenAI specification.
Currently, KubeAI extension of the platform for FasterWhisper audio models supports serving the model from
Huggingface only. 

For this to work the Whisper model trained using the procedures present in this project should be converted to FasterWhisper-compatible
representation and uploaded to HuggingFace.

## 1. Convert the model to FasterWhisper format

The following snippet used to download the model to the local execution environment.
```python
model = project.get_model("whisper-ft")
model.download("./model/whisper-ft", overwrite=True)
```

Install the necessary dependencies:

```shell
pip install faster-whisper transformers torch==2.8.0
```

```python
from ctranslate2.converters import TransformersConverter

tc = TransformersConverter("./model/whisper-ft", copy_files=['tokenizer.json', 'preprocessor_config.json'])
tc.convert('./model/my-audio-model', quantization="float16")
```

Please ensure that the README file is present in the model folder. The model should have ``ctranslate2`` as its ``library_name`` and
``automatic-speech-recognition`` in its tags, e.g.,:

```yaml
tags:
  - audio
  - automatic-speech-recognition
license: mit
library_name: ctranslate2
...
```

## 2. Upload the model to HugginFace

Follow the standard procedure to upload model to HuggingFace.  See [here](https://huggingface.co/docs/hub/en/models-uploading) for
complete instructions on how to perform upload procedure using different tools.

## 3. Explose the FasterWhisper model

It is possible to use FasterWhipser model as is, servinig it using the KubeAI platform component.

```python
audio_function = project.new_function("audio",
                                    kind="kubeai-speech",
                                    model_name="audiomodel",
                                    url="hf://myorg/my-audio-model")


run = audio_function.run(action="serve") 
MODEL_ID = run.refresh().status.openai['model']
```

Once deployed, the model is available and it is possible to call the OpenAI-compatible API from within the platform (/openai/v1/transcriptions endpoint):

```python
from openai import OpenAI

client = OpenAI(base_url=f"http://{KUBEAI_ENDPOINT}/openai/v1", api_key="ignore")
audio_file= open("audio.wav", "rb")

transcription = client.audio.transcriptions.create(
    model=MODEL_ID, 
    file=audio_file
)

print(transcription.text)
```

