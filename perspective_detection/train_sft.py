from transformers import AutoModelForCausalLM, AutoTokenizer,HfArgumentParser,Seq2SeqTrainingArguments
import wandb
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.file_utils import read_tsv_file
from functools import partial
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model

from accelerate import PartialState


from dataclasses import dataclass, field
from typing import Optional

instruction_template = "[INST]"
response_template = "[/INST]"

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_loftq: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables LoftQ init for the LoRA adapters when using QLoRA."},
    )
    use_loftq_callback: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables LoftQ callback comparing logits of base model to the ones from LoftQ init. Provides better init."},
    )
    save_model: Optional[str] = field(
        default="model",
        metadata={"help": "Path to save the model."},
    )
    
def formulate_text(instruction, doc_1, doc_2):
    prompt = instruction.replace('[Doc 1]', doc_1).replace('[Doc 2]', doc_2)
    return prompt

    
def gen_data(file_name):
    data = read_tsv_file(file_name)[1:]
    for row in data:
        input_text = formulate_text(instruction, row[2], row[3])
        label = 'Yes' if bool(str(row[4]) == 'True') else 'No'
        txt = f"[INST] {input_text} [/INST]{label}</s> "
        yield {"text":  txt}
    


def load_model(base_model, device_string):
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map={'':device_string})
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


# parse arguments
parser = HfArgumentParser(
        (ModelArguments, Seq2SeqTrainingArguments)
    )

model_args, training_arguments = parser.parse_args_into_dataclasses()


# Set device
device_string = PartialState().process_index


#Use a sharded model to fine-tune in the free version of Google Colab.
base_model = "mistralai/Mistral-7B-Instruct-v0.2" 
instruction = open("instructions/no-chat/instructions.txt", "r").read()

# load model
model, tokenizer = load_model(base_model, device_string)  
peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
        )
 
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load the dataset
generator = partial(gen_data, "train_data.tsv")
ds = Dataset.from_generator(generator)
tokenized_data = ds.train_test_split(test_size=0.1, seed=42)


# init wandb
wandb.login(key = "0ef3469b8fbbb3d523bfe2ec43bfd0408e681227")
run = wandb.init(project='DiverseRetrieval', job_type="training", anonymous="allow", group='group_name')


training_arguments.gradient_checkpointing_kwargs = {'use_reentrant':False}
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    args=training_arguments,
    peft_config=peft_config,
    dataset_text_field="text",
    data_collator=collator,
    packing=False,
    max_seq_length=2048
)


# training
trainer.train() 
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

# Save the fine-tuned model
trainer.save_model(model_args.save_model)
wandb.finish()