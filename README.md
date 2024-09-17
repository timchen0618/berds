# BERDS: A Benchmark for Retrieval Diversity for Subjective Questions

This is the repository that contains source code for the [BERDS website](https://timchen0618.github.io/berds/).


## Requirements 
Tested on Python 3.8.  
To use the repo, first clone the project. 
```shell
git clone git@github.com:timchen0618/berds.git
```

And create a virtual environment (recommended).  
```shell
cd berds/
python3 -m venv berds
source berds/bin/activate
```

### Environment
```
pip install -r requirments.txt
```

### Data & Model
You can find the data and model [here](https://huggingface.co/collections/timchen0618/berds-66e8a20cd683a3f4e54d0b62).  

#### Download Data
You can load the data from huggingface, and later save it if needed.  
```python
from datasets import load_dataset

arguana_ds = load_dataset("timchen0618/Arguana")
kialo_ds = load_dataset("timchen0618/Kialo")
opinionqa_ds = load_dataset("timchen0618/OpinionQA")

```

#### Load Model
You can load the model from huggingface, with the `peft` and `transformers` libraries.  
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("timchen0618/Mistral_BERDS_evaluator")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "timchen0618/Mistral_BERDS_evaluator")
```


## Perspective Detection
Given a documents and a perspective, **perspective detection** is defined as "identifying whether the document supports or implies the perspective".  
A perspective detection model is an essential component of the automatic evaluation.  

More details on this can be found [here](perspective_detection/README.md). 


## Reproduction
More on this soon. 

## Evaluate Retrieval Outputs
### Expected Format

### Commands

