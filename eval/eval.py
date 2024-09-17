from Eval.prediction_mistral import Prediction
from utils.model_utils import load_model
from utils.file_utils import read_jsonl, write_jsonl

import random
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

import torch
import transformers

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def mrecall(preds):
    mrecalls = []
    topk = len(preds[0][0])
    for inst in preds:
        if len(inst) > topk:
            mrecalls.append(int(sum([any(preds_per_perspective) for preds_per_perspective in inst])>=topk))
        else:
            mrecalls.append(int(all([any(preds_per_perspective) for preds_per_perspective in inst])))
        
    return sum(mrecalls) / float(len(mrecalls))
          

def precision(preds):
    precisions = []
    
    topk = len(preds[0][0])
    for inst in preds:
        assert len(inst[0]) >= topk
        num_perspective_containing_docs = 0
        for j in range(topk):
            contain_any_perspective = False
            for p in inst:
                if p[j]:
                    contain_any_perspective = True
                    break
            if contain_any_perspective:
                num_perspective_containing_docs += 1
            
        precisions.append(num_perspective_containing_docs / topk)
        
    return sum(precisions) / float(len(precisions))

def prepare_predictor_and_instruction(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # read instructions
    instruction = open(args.instructions).read()

    # load models     
    assert args.model.lower().find(args.model_type) != -1
    if args.model.find('llama') != -1: # llama models
        model, configs = load_model(args, device, logger)
        tokenizer = None
    elif args.model != 'gpt4':         # t5 models
        model, tokenizer = load_model(args, device, logger)
        model.eval()
    else:                              # gpt4 models
        model = None
        tokenizer = None

    predictor = Prediction(args.model_type, model, tokenizer, device, args)
    return instruction, predictor

def main(args):
    random.seed(0)
    torch.manual_seed(0)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    
    preds = []
    if not args.compute_only:
        """
            Running perspective detection module, and store the results in args.output_file
            
            Data Format (each instance):
            {
                "question": ...,
                "ctxs": [
                    {"title": ..., "text": ...},
                    {"title": ..., "text": ...},
                    ...
                ],
                "perspectives": [
                    "perspective 1",
                    "perspective 2",
                    ...
                ]
            }
            
            After running the perspective detection module, the "ctxs" field of the output file will have the following format:
            "ctxs": [
                {"title": ..., "text": ..., "mistral-preds": [True, False, ...]},
                {"title": ..., "text": ..., "mistral-preds": [True, False, ...]},
                ...
            ],
        """
        
        logger.info('running perspective detection: eval')
        assert Path(args.data).suffix == '.jsonl'
        
        # prepare predictor 
        instruction, predictor = prepare_predictor_and_instruction(args)
        
        # prepare data
        data = read_jsonl(args.data)

        for inst in tqdm(data):
            pred_inst = []  # (num_perspectives, num_docs)
            # retrieving perspectives
            perspectives = inst['perspectives']
            
            # retrieving docs
            docs = inst['ctxs']
                
            # for every perspective, check if it is supported by the docs
            for p in perspectives:
                pred_inst.append([])
                for doc in docs[:args.topk]:
                    if 'title' in doc:
                        doc_text = doc['text'] + ' ' + doc['title']
                    else:
                        doc_text = doc['text']   
                        
                    pred = predictor.make_prediction(p, doc_text, instruction)  # whether each perspective is supported by doc_text
                    if args.model_type == 'gpt4':
                        if 'gpt4-preds' not in doc:
                            doc['gpt4-preds'] = []
                            
                        if len(doc['gpt4-preds']) <= len(perspectives):
                            doc['gpt4-preds'].append(pred)
                    elif args.model_type == 'mistral':
                        if 'mistral-preds' not in doc:
                            doc['mistral-preds'] = []
                            
                        if len(doc['mistral-preds']) <= len(perspectives):
                            doc['mistral-preds'].append(pred)
                    else:
                        raise NotImplementedError

                    pred_inst[-1].append(pred)
            preds.append(pred_inst)    
        
        write_jsonl(args.output_file, data)
    else:
        """
            Not running perspective detection module, only computing scores based on previously perspective detection results
        """
        assert (Path(args.data).name.find('.gpt4pred') != -1 or Path(args.data).name.find('.mistralpred') != -1)
        logger.info("only computing scores based on previously perspective detection results")
        data = read_jsonl(args.data)                   
        
        for inst in tqdm(data):
            pred_inst = []  # (num_perspectives, num_docs)
            perspectives = inst['perspectives']
            docs = inst['ctxs']

            pred_str = 'gpt4-preds' if Path(args.data).name.find('.gpt4pred') != -1 else 'mistral-preds'
            for j, p in enumerate(perspectives):
                pred_inst.append([])
                for doc in docs[:args.topk]:
                    pred = doc[pred_str][j]
                    pred_inst[-1].append(pred)

            preds.append(pred_inst)
            
    # compute Precision & MRecall
    precision_score = precision(preds)
    mrecall_score = mrecall(preds)
    
    logger.info(f'average precision: {100*precision_score:.2f}')
    logger.info(f'mrecall: {100*mrecall_score:.2f}')
    
    # writing score to a csv file
    fw = open('score.csv', 'w')
    fw.write(f'\n{100*precision_score:.2f}\n{100*mrecall_score:.2f}')
    fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5_xxl_true_nli_mixture", type=str)
    parser.add_argument("--model_type", default="t5", type=str)
    parser.add_argument("--output_file", type=str, default="output.jsonl.mistralpred")
    parser.add_argument("--instructions", type=str, default="instructions_nli.txt")
    parser.add_argument("--topk", type=int, default=10)  
    parser.add_argument("--compute_only", action='store_true') 
    
    parser.add_argument("--data", type=str, default="/path/to/some/retrieval/output.jsonl")
    args = parser.parse_args()

    main(args)
    
