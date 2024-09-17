from utils.file_utils import collect_retrieval_results
import logging
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from angle_emb import AnglE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DocSimilarity:
    def __init__(self, model):
        self.model = model
        self.saved_similarities = [] # list of document similarities (100*100 matrices)
    
    @torch.no_grad()
    def similarity(self, documents):
        self.saved_similarities.append(np.zeros((100, 100)))
        all_vecs = self.model.encode(documents, to_numpy=False)
        for i in range(len(documents)-1):
            target_vec = all_vecs[i:i+1]
            vecs = all_vecs[i+1:]
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            output = cos(target_vec, vecs) # (len(documents)-i-1)
            self.saved_similarities[-1][i, i+1:len(documents)] = output.cpu().numpy()
            
    def clear_similarity(self):
        self.saved_similarities = []

    
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)
    
    # obtain retrieval results 
    data_types = ['arguana_generated', 'kialo', 'opinionqa']
    retrieval_results, rootdir = collect_retrieval_results(args.corpus, args.retriever, data_types)
    
    doc_sim = DocSimilarity(model)
    data_type_id = 0
    for retrieval_results_per_type in retrieval_results:
        for inst in tqdm(retrieval_results_per_type):
            documents = []
            for doc in inst['ctxs']:
                # get doc_text from documents 
                if 'title' in doc:
                    doc_text = doc['text'] + ' ' + doc['title']
                elif 'wikipedia_title' in doc:
                    doc_text = doc['text'] + ' ' + doc['wikipedia_title']
                else:
                    doc_text = doc['text']
                documents.append(doc_text)
            # compute simliarity between documents
            doc_sim.similarity(documents)
        
        # save the simliarity matrix
        doc_sim.saved_similarities = [sim.reshape(-1, sim.shape[1], sim.shape[1]) for sim in doc_sim.saved_similarities]
        similarities = np.concatenate(doc_sim.saved_similarities, axis=0)
        
        # make the similarity matrix symmetric
        for i in range(similarities.shape[0]):
            for j in range(similarities.shape[1]-1):
                for k in range(j+1, similarities.shape[1]):
                    similarities[i, k, j] = similarities[i, j, k]
        
        # save the similarity matrix
        np.save(f'{rootdir}/{data_types[data_type_id]}_similarities.npy', similarities)
        doc_sim.clear_similarity()  # clear the saved similarities
        data_type_id += 1

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default='wiki')
    parser.add_argument("--retriever", type=str, default='dpr')
    args = parser.parse_args()
    
    main(args)
    
    """
        Example usage:
        python gen_doc_similarity.py --corpus wiki --retriever dpr
        
        The script will compute the similarity between documents in the retrieval results.
        The similarity matrix will be saved in the same directory as the retrieval results.
        You will later use reraanking.py to rerank the retrieval results based on the saved similarity matrix.
    """