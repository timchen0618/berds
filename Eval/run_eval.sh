#!/bin/bash
for corpus_and_retriever in sphere,bm25 
do
    IFS=","
    set -- ${corpus_and_retriever}
    echo "corpus: $1 | retriever: $2"
    ROOT="/path/to/retrieval_outputs/${1}/${2}"
    PORT=29500
    TOPK=5
    
    for DATA in "arguana_generated.jsonl" "kialo.jsonl" "opinionqa.jsonl" 
    do
        MODEL_SHORT="mistral" 
        MODEL_NAME="timchen0618/Mistral_BERDS_evaluator"
        OUTPUT="${ROOT}/${DATA}.${MODEL_SHORT}pred"

        PYTHONPATH=.. torchrun --nproc_per_node 1 --master-port ${PORT} eval.py \
                --data ${ROOT}/${DATA} \
                --output_file ${OUTPUT}   \
                --instructions instructions_chat.txt \
                --model ${MODEL_NAME}  \
                --model_type ${MODEL_SHORT} \
                --topk ${TOPK} 
    done 
done
        



         



