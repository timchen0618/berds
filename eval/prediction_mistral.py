import json
import torch
import openai
from functools import partial
from pathlib import Path
import ast
from transformers import pipeline

class Prediction:
    def __init__(self, model_type, model, tokenizer, device, args):
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        self.args = args
    
    def formulate_text(self, instruction, doc_1, doc_2):
        prompt = instruction.replace('[Doc 1]', doc_1).replace('[Doc 2]', doc_2)
        return prompt

    def formulate_dialogs(self, instruction, doc_1, doc_2):
        start_tok = '<|im_start|>'
        end_tok = '<|im_end|>'
        instruction = instruction.replace('[Doc 1]', doc_1).replace('[Doc 2]', doc_2)
        messages = instruction.split(end_tok)[:-1]
        messages = [{"role": m.strip('\n').split('\n')[0][12:], "content": '\n'.join(m.strip('\n').split('\n')[1:]).strip()} for m in messages]    
        return messages
        

    def make_prediction(self, hypothesis, premise, instruction_nli):
        chat = Path(self.args.instructions).stem.find('chat') != -1
        if chat:
            input_text = self.formulate_dialogs(instruction_nli, premise, hypothesis)
        else:
            input_text = self.formulate_text(instruction_nli, premise, hypothesis)
        
        response = self.inference(self.model, self.tokenizer, input_text, self.device, self.model_type, chat)

        # separate response for Mistral
        if self.model_type == 'mistral':
            instruction_sep = '[/INST]'
            response = self.sep_response_from_pred(response, instruction_sep)
            
        return self.parse_response(response) 
            

    @torch.no_grad()
    def inference(self, model, tokenizer, input_text, device, model_type, chat):
        # produce an output given the input_text
        assert (model_type == 'mistral' or model_type == 'gpt4')
        assert chat
        if model_type == 'gpt4':
            return self.pred_gpt4(input_text, chat)
        elif model_type == 'mistral':
            return self.pred_mistral(input_text, tokenizer, model, device)
        else:
            raise NotImplementedError


    @torch.no_grad()
    def pred_mistral(self, input_text, tokenizer, model, device):
        model.eval()
        model_inputs = tokenizer.apply_chat_template(input_text[1:], return_tensors="pt").to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=16, do_sample=False)
        raw_response = tokenizer.batch_decode(generated_ids)[0]
        return raw_response.strip('\n').strip().split('</s>')[0]
    
    @torch.no_grad()
    def pred_gpt4(self, prompt, chat=False):
        if chat:
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
        )
        response = response['choices'][0]['message']['content']
        return response

    def sep_response_from_pred(self, response, instruction_sep):
        response = response.split(instruction_sep)[1].strip()
        return response


    def parse_response(self, response):
        if response[0] != '{':
            response = response.strip().strip('\n').split('Answer:')[-1].split('The answer is')[-1].split('The answer is \"')[-1].split('The answer is:')[-1].split('Based on the information provided in the document, the answer is')[-1].split('Based on the information provided in the document, the answer is:\n')[-1].strip()
            response = response.strip().lower()
            _yes = (response[:3] == 'yes' or response[:4] == '(yes') or (response == 'y' or response == 'ye') or (response[-3:] == 'yes' or response[-4:] == 'yes.' or response[-5:] == 'yes\".')
            _entail_or_true = (response[:6] == 'entail' or (response[:4] == 'true'))
            return  _yes or _entail_or_true
        else:
            return ast.literal_eval(response)['Answer'].strip().lower() == 'yes'
        