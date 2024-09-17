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
        if self.args.model == 'google/t5_xxl_true_nli_mixture':
            input_text = "premise: %s hypothesis: %s"%(premise[:2000], hypothesis)
            return self.pred_decoder_models(input_text, self.tokenizer, self.model, self.device)
        else:
            chat = Path(self.args.instructions).stem.find('chat') != -1
            if chat:
                input_text = self.formulate_dialogs(instruction_nli, premise, hypothesis)
            else:
                input_text = self.formulate_text(instruction_nli, premise, hypothesis)
            
            response = self.inference(self.model, self.tokenizer, input_text, self.device, self.model_type, chat)
            # gives you the whole conversation - gemma, mistral
            # gives you the response - gpt4, llama, zephyr
            # chat models: gpt4, llama, zephyr
            # non-chat models: gpt4, llama, mistral, gemma
            
            if self.model_type in ['mistral', 'gemma']:
                if self.model_type == 'mistral':
                    instruction_sep = '[/INST]'
                else:
                    instruction_sep = instruction_nli.strip('\n').split('\n')[-1]
                response = self.sep_response_from_pred(response, instruction_sep, self.model_type)
                
            return self.parse_response(response, self.model_type) 
            

    @torch.no_grad()
    def inference(self, model, tokenizer, input_text, device, model_type, chat):
        # produce an output given the input_text
        # chat models: gpt4, llama, zephyr
        # non-chat models: gpt4, llama, mistral, gemma
        if model_type == 'gpt4':
            return self.pred_gpt4(input_text, chat)
        elif model_type == 'llama':
            return self.pred_llama(model, input_text, chat)
        elif model_type == 'mistral':
            assert chat
            return self.pred_mistral(input_text, tokenizer, model, device)
        elif model_type == 'gemma':
            assert not chat
            return self.pred_gemma(input_text, tokenizer, model, device)
        elif model_type == 'zephyr':
            assert chat
            return self.pred_zephyr(model, input_text)


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

    @torch.no_grad()
    def pred_llama(self, model, prompt, chat=False):
        assert chat
        dialogs = [prompt] if chat else []
        results = model.chat_completion(
            dialogs,  
            max_gen_len=512,
            temperature=0.6,
            top_p=0.9,
        )
        response = results[0]['generation']['content'].strip('\n').strip()
        return response

    @torch.no_grad()
    def pred_mistral(self, input_text, tokenizer, model, device):
        model.eval()
        model_inputs = tokenizer.apply_chat_template(input_text[1:], return_tensors="pt").to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=16, do_sample=False)
        raw_response = tokenizer.batch_decode(generated_ids)[0]
        return raw_response.strip('\n').strip().split('</s>')[0]

    def pred_gemma(self, input_text, tokenizer, model, device):
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(**input_ids, max_new_tokens=512)
        return tokenizer.decode(outputs[0]).strip('\n').strip('<bos>').strip('<eos>').strip()

    @torch.no_grad()
    def pred_zephyr(self, model, messages):
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return model(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)[0]["generated_text"].split('<|assistant|>')[-1]

    def sep_response_from_pred(self, response, instruction_sep, model_type):
        assert model_type in ['gemma', 'mistral']
        response = response.split(instruction_sep)[1].strip()
        return response


    def parse_response(self, response, model_type):
        if response[0] != '{':
            response = response.strip().strip('\n').split('Answer:')[-1].split('The answer is')[-1].split('The answer is \"')[-1].split('The answer is:')[-1].split('Based on the information provided in the document, the answer is')[-1].split('Based on the information provided in the document, the answer is:\n')[-1].strip()
            response = response.strip().lower()
            _yes = (response[:3] == 'yes' or response[:4] == '(yes') or (response == 'y' or response == 'ye') or (response[-3:] == 'yes' or response[-4:] == 'yes.' or response[-5:] == 'yes\".')
            _entail_or_true = (response[:6] == 'entail' or (response[:4] == 'true'))
            return  _yes or _entail_or_true
        else:
            return ast.literal_eval(response)['Answer'].strip().lower() == 'yes'
        


    @torch.no_grad()
    def pred_decoder_models(self, input_text, tokenizer, model, device):
        model.eval()
        input_ids = tokenizer(
            input_text, padding=True, truncation=True, return_tensors="pt", max_length=256
        ).input_ids.to(device)
        decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids.to(device)[:, :1]

        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        all_logits = torch.cat((outputs['logits'][:, 0, 3:4], outputs['logits'][:, 0, 209:210]), dim=-1)
        probs = torch.softmax(all_logits, dim=-1)
        
        input_ids = input_ids.detach().cpu()
        decoder_input_ids = decoder_input_ids.detach().cpu()
        probs = probs.detach().cpu()
        
        pred_instance = torch.count_nonzero((probs[:, 1] > self.args.threshold).long()).item() > 0
        return pred_instance