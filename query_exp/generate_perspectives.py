import openai
import json
import csv
from tqdm import tqdm

def pred_gpt4(prompt):
    messages = [{"role": "user", "content": prompt}]
        
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
    )
    response = response['choices'][0]['message']['content']
    return response

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')

def form_prompt(instruction, input_text):
    return instruction.replace('[Claim]', input_text).replace('[Question]', input_text)

def write_tsv(filename, data):
    with open(filename, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerows(data)



rootdir = '../Data/'
folder_map = {"arguana_generated": "Arguana", "kialo": "Kialo", "opinionqa": "OpinionQA"}
instruction = open('instruction.txt', 'r').read()

for data_type in ['arguana_generated', 'kialo', 'opinionqa']:
    perspectives = []
    data = read_jsonl(f'{rootdir}/{folder_map[data_type]}/{data_type}.test.jsonl')
    for inst in tqdm(data):
        out_inst = []
        prompt = form_prompt(instruction, inst['question'])
        try:
            response = pred_gpt4(prompt)
        except openai.error.APIError:
            print('skipped question', inst['question'])
            continue
        
        try:
            perspectives_dict = json.loads(response)
            perspectives.append({"question": inst['question'], "perspectives":[{"name":k, "text":v} for k, v in perspectives_dict.items()]})
        except:
            perspectives.append({"question": inst['question'], "perspectives":response})


    new_data = []
    assert len(data) == len(perspectives)
    _id = 0
    for i in range(len(data)):
        assert data[i]['question'] == perspectives[i]['question']
        for p in perspectives[i]['perspectives']:
            
            if type(p['text']) == dict:
                p['text'] = list(p['text'].values())[0]
            new_data.append({
                'org_id': data[i]['id'],
                'id': _id,
                'org_q': data[i]['question'],
                'perspective': p['name'],
                'text': p['text'],
                'question': data[i]['question'] + ' ' + p['text'],
                'input': data[i]['question'] + ' ' + p['text']
            })
            _id += 1
            
    write_jsonl(f'data/{data_type}_query_exp.jsonl', new_data)