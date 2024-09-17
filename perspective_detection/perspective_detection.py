import logging
import argparse
import torch
from tqdm import tqdm
import random
from prediction import Prediction
import os
from utils.model_utils import load_model
from utils.file_utils import read_tsv_file, write_tsv, append_tsv

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(preds, labels):
    return sum([int(preds[i] == labels[i]) for i in range(len(preds))]) / len(preds)

def f1(preds, labels):
    tp = sum([int(preds[i] == labels[i] == 1) for i in range(len(preds))])
    fp = sum([int(preds[i] == 1 and labels[i] == 0) for i in range(len(preds))])
    fn = sum([int(preds[i] == 0 and labels[i] == 1) for i in range(len(preds))])
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)
    
    # read instructions
    instruction = open(args.instructions).read()
    # load models     
    assert args.model.lower().find(args.model_type) != -1
    if args.model.find('llama') != -1: # llama models
        model, configs = load_model(args, device, logger)
        tokenizer = None
    elif args.model != 'gpt4':         # other models
        model, tokenizer = load_model(args, device, logger)
    else:                              # gpt4 models
        model = None
        tokenizer = None

    
    labels = []
    preds = []
    predictor = Prediction(args.model_type, model, tokenizer, device, args)
    
    data = read_tsv_file(args.data)
    for inst in tqdm(data):
        if inst[0] == 'question':
            continue
        document = inst[2]
        aspect = inst[3]
        label = int(inst[4])
        labels.append(label)
        
        hypothesis = aspect
        premise = document
        
        pred = predictor.make_prediction(hypothesis, premise, instruction)
            
        preds.append(int(pred))
        if args.write_results:
            inst.append(str(int(pred)))
        
    _acc = accuracy(preds, labels)  
    _f1 = f1(preds, labels)                         
    logger.info(f'Acc: {_acc}')
    logger.info(f'F1: {_f1}')           
    if args.write_results:
        logger.info(f'writing outputs to {args.output_file}')
        write_tsv(args.output_file, data)
    
    if args.write_scores:
        model_str = args.model.split('/')[-1]
        score_file = f'scores/test/score_{model_str}.tsv'
        logger.info(f'writing scores to {score_file}')
        
        if not os.path.exists(score_file):
            append_tsv(score_file, [['instruction', 'Acc', 'F1']])
        inst_str = args.instructions.split('_nli_')[-1].split('.txt')[0]
        logger.info(inst_str)           
        append_tsv(score_file, [[inst_str, '%2.2f'%(100*_acc), '%2.2f'%(100*_f1)]])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5_xxl_true_nli_mixture", type=str)
    parser.add_argument("--model_type", default="t5", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--output_file", type=str, default="results_GPT4.csv")
    parser.add_argument("--saved_model_name", type=str, default="saved_models")
    parser.add_argument("--instructions", type=str, default="instructions.txt")
    parser.add_argument("--write_results", action='store_true')
    parser.add_argument("--write_scores", action='store_true')
    parser.add_argument("--llama_config", type=str, default="configs/llama-chat-config.yaml")
    
    parser.add_argument("--data", type=str, default="/scratch/cluster/hungting/projects/Multi_Answer/Sphere/output/sparse/opinion_selected.jsonl")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    main(args)

