#!/bin/bash
for corpus_and_retriever in sphere,bm25 
do
    IFS=","
    set -- ${corpus_and_retriever}
    echo "corpus: $1 | retriever: $2"
    ROOT="/scratch/cluster/hungting/projects/Multi_Answer/Data/retrieval_outputs/${1}/${2}"
    PORT=29600
    TOPK=5
    
    for DATA in "arguana_generated_1k.jsonl" "kialo_1k.jsonl" "opinionqa_1k.jsonl" 
    do
        MODEL_SHORT="mistral" 
        MODEL_NAME="/scratch/cluster/hungting/projects/Multi_Answer/Subtask_1/saved_models/Mistral_BERDS_evaluator_full"
        OUTPUT="${ROOT}/${DATA}.${MODEL_SHORT}pred"

        PYTHONPATH=.. python eval_vllm.py \
                --data ${ROOT}/${DATA} \
                --output_file ${OUTPUT}   \
                --instructions instructions_chat.txt \
                --model ${MODEL_NAME}  \
                --model_type ${MODEL_SHORT} \
                --topk ${TOPK} 
    done 
done
        



         



