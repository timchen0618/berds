from utils.file_utils import collect_retrieval_results, write_jsonl
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_DOCS=100
corpus = 'google_api' # ['wiki', 'sphere', 'arguana']
retriever = 'bm25' # ['contriever', 'google', 'dpr', 'bm25']
data_types = ['arguana_generated', 'kialo', 'opinionqa']

raw_results, rootdir = collect_retrieval_results(corpus, retriever, data_types, query_exp=False, rerank=False)
qe_results, _ = collect_retrieval_results(corpus, retriever, data_types, query_exp=True, rerank=False)

assert len(raw_results) == len(qe_results), (len(raw_results), len(qe_results))
assert len(raw_results) == len(data_types), (len(raw_results), len(data_types))
    
for data_type, data, qe_data in zip(data_types, raw_results, qe_results):
    logger.info(f'Processing {data_type}')
    query2doc_sets = {}  # query mapped to a set of docs, where each set come from a different query expansion 
    query2docs = {}      # the actual container for the docs, combining the sets from query2doc_sets (from each qe results)
    for inst in qe_data: # group docs based on question
        assert 'ctxs' in inst
        if inst['org_q'] not in query2doc_sets:
            query2doc_sets[inst['org_q']] = []
        query2doc_sets[inst['org_q']].append(inst['ctxs'])
        
    for k, v in query2doc_sets.items():  # take turns taking top documents from each qe set
        query2docs[k] = []
        for i in range(len(v[0])):
            for docs in v:
                query2docs[k].append(docs[i])
            if len(query2docs[k]) >= NUM_DOCS:
                query2docs[k] = query2docs[k][:NUM_DOCS]
                break
            
        assert len(query2docs[k]) == NUM_DOCS

    for inst in data:
        inst['ctxs'] = query2docs[inst['question']]
    
    write_jsonl(f'{rootdir}/{data_type}_query_exp_processed.jsonl', data)
    logger.info(f'Finished processing {data_type}')