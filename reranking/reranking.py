from utils.file_utils import collect_retrieval_results
import logging
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')
            

class Reranker:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        self.args = args
        self.similarity_matrix = None
        self._lambda = args._lambda
        max_score_map = {'wiki': {"bm25":63.5, "dpr":100.6, "contriever":2.20}, 'sphere':{"bm25":99.9, "dpr":243.9, "contriever":2.55}, 'google_api':{"bm25":181.5, "contriever":2.52}}
        self.max_score = max_score_map[args.corpus][args.retriever]

    def rerank(self, retrieval_results):
        assert 'ctxs' in retrieval_results[0]
        
        # compute max score
        self.max_scores = []
        for inst in retrieval_results:
            max_score = 0
            for doc in inst['ctxs']:
                max_score = max(max_score, float(doc['score']))
            self.max_scores.append(max_score)
        print('max_scores:', self.max_scores)
        
        data_idx = 0
        for inst in tqdm(retrieval_results):
            documents = inst['ctxs']
            new_document_ids_n_scores = [(0, float(documents[0]['score']))]
            for _ in range(len(documents) - 1):
                new_document_id, mmr_score = self.add_one_document(documents, new_document_ids_n_scores, data_idx)
                new_document_ids_n_scores.append((new_document_id, mmr_score))
            inst['ctxs'] = []
            
            for id_n_score in new_document_ids_n_scores:
                doc = documents[id_n_score[0]]
                doc['score'] = id_n_score[1]
                inst['ctxs'].append(doc)
                
            data_idx += 1
        
        return retrieval_results
    
    def load_similarity_matrix(self, similarity_matrix_path):
        self.similarity_matrix = np.load(similarity_matrix_path)

    def add_one_document(self, documents, retrieved_document_ids_n_scores, data_idx):
        mmr = -100
        new_document_id = 0
        retrieved_document_ids = [i for i, _ in retrieved_document_ids_n_scores]
        for i in range(len(documents)):
            if i not in retrieved_document_ids:
                # mmr_i = self._lambda * float(documents[i]['score']) / self.max_score
                max_score = self.max_scores[data_idx] if self.max_scores[data_idx] != 0 else self.max_score
                mmr_i = self._lambda * float(documents[i]['score']) / max_score
                
                # compute maximum similarity 
                max_sim_between_docs = -100
                for j in retrieved_document_ids:
                    assert i != j
                    sim_i_j = self.similarity_matrix[data_idx][i][j]
                    if sim_i_j > max_sim_between_docs:
                        max_sim_between_docs = sim_i_j
                 
                mmr_i -= (1 - self._lambda) * max_sim_between_docs
        
                if mmr_i > mmr:
                    mmr = mmr_i
                    new_document_id = i
        return new_document_id, mmr

    def similarity(self, i, j):
        assert i != j
        return self.similarity_matrix[i, j]   
    
def compute_max_score(retrieval_results):
    max_score = 0
    avg_scores = []
    for results in retrieval_results:
        for inst in results:
            for doc in inst['ctxs']:
                max_score = max(max_score, float(doc['score']))
                avg_scores.append(float(doc['score']))
    return max_score, np.mean(np.array(avg_scores))


def main(args):
    data_types = ['arguana_generated', 'kialo', 'opinionqa']
    logger.info('collecting retrieval results')
    retrieval_results, rootdir = collect_retrieval_results(args.corpus, args.retriever, data_types)
    
    logger.info(f"Reranking results for {args.corpus} corpus using {args.retriever} retriever, lambda={args._lambda}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = None, None
    
    reranker = Reranker(tokenizer=tokenizer, model=model, device=device, args=args)
    data_type_id = 0
    for retrieval_results_per_type in retrieval_results:
        reranker.load_similarity_matrix(f'{rootdir}/{data_types[data_type_id]}_similarities.npy')
        reranked_results = reranker.rerank(retrieval_results_per_type)
        write_jsonl(f'{rootdir}/{data_types[data_type_id]}_1k_reranked_l{args._lambda}.jsonl', reranked_results)
        data_type_id += 1
                              
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default='wiki')
    parser.add_argument("--retriever", type=str, default='dpr')
    parser.add_argument("--_lambda", type=float, default=0.5)
    args = parser.parse_args()
    
    main(args)
    
    """
        Before running this script, you need to run gen_doc_similarity.py to generate the similarity matrix. 
        You will then compute reranking based on the saved similarity matrix.
    """
