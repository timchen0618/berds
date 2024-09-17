#!/bin/bash
for corpus_and_retriever in google_api,bm25 
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
        MODEL_NAME="/path/to/mistral/model"
        OUTPUT="${ROOT}/${DATA}.${MODEL_SHORT}pred"

        PYTHONPATH=.. torchrun --nproc_per_node 1 --master-port ${PORT} eval.py \
                --data ${ROOT}/${DATA} \
                --output_file ${OUTPUT}   \
                --instructions instructions_chat.txt \
                --model ${MODEL_NAME}  \
                --model_type ${MODEL_SHORT} \
                --topk ${TOPK} \
                --test_data_only
    done 
done
        



         



