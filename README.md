# BERDS: A Benchmark for Retrieval Diversity for Subjective Questions

This is the repository that contains source code for the [BERDS website](https://timchen0618.github.io/berds/).


## Requirements 
### Environment
```
pip install -r requirments.txt
```

### Data & Model
You can find the data and model [here](https://huggingface.co/collections/timchen0618/berds-66e8a20cd683a3f4e54d0b62).  

#### Download Data
You can load the data from huggingface, and later save it if needed.  
```
from datasets import load_dataset

arguana_ds = load_dataset("timchen0618/Arguana")
kialo_ds = load_dataset("timchen0618/Kialo")
opinionqa_ds = load_dataset("timchen0618/OpinionQA")

```

#### Load Model
You can load the model from huggingface, with the `peft` and `transformers` libraries.  
```
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("timchen0618/Mistral_BERDS_evaluator")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "timchen0618/Mistral_BERDS_evaluator")
```


## Perspective Detection

## Reproduction

## Evaluate Retrieval Outputs


