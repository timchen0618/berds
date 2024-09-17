import json

def read_tsv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
    return data

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

import csv

def write_tsv(filename, data):
    with open(filename, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerows(data)
        
def append_tsv(filename, data):
    with open(filename, 'a', newline='') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerows(data)

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')
            

def process_items(string, _strip=False):
    
    for i in range(20):
        string = string.replace('\"\"', '\"')
    if _strip:
        string = string.strip('\"')
        
    return string




def collect_retrieval_results(corpus, retriever, data_types, query_exp=False, rerank=False, syco=False, syco_pos=False, syco_neg=False):
    ROOT='/path/to/retrieval/outputs'
    
    if query_exp and rerank:
        raise ValueError("Cannot have both query_exp and rerank")
    all_data = []
    if retriever in ['contriever', 'dpr', 'bm25', 'google_api']:
        rootdir = f'{ROOT}/{corpus}/{retriever}'
    else:
        raise ValueError(f"Retriever {retriever} not supported")
        
    for data_type in data_types:
        if query_exp:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}_query_exp.jsonl')) 
        elif rerank:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}_reranked.jsonl')) 
        elif syco:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}_syco.jsonl'))
        elif syco_pos:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}_syco_pos.jsonl'))
        elif syco_neg:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}_syco_neg.jsonl'))          
        else:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}.jsonl'))
    return all_data, rootdir