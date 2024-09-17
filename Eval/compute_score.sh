#!/bin/bash

all_string=""

for corpus_and_ret in wiki,bm25 wiki,dpr wiki,contriever
do
    IFS=","
    set -- ${corpus_and_ret}
    echo "corpus: $1 | retriever: $2"
    ROOT="/path/to/retrieval_outputs/${1}/${2}"
    PORT=29500
    TOPK=5
    csv_string=""

    for DATA in "arguana_generated" "kialo" "opinionqa" 
    do
        DATA="${DATA}_1k${SUFFIX}.jsonl"
        for MODEL_SHORT in "mistral" 
        do
            MODEL_SHORT="mistral" 
            MODEL_NAME="/path/to/mistral/model"
            INPUT="${ROOT}/${DATA}.${MODEL_SHORT}pred"
            
            PYTHONPATH=.. torchrun --nproc_per_node ${NUM_NODE} --master-port ${PORT} eval.py \
                    --data ${INPUT} \
                    --output_file  foo.txt  \
                    --instructions instructions.txt \
                    --model ${MODEL_NAME}  \
                    --model_type ${MODEL_SHORT} \
                    --topk $TOPK --compute_only
                
            _STR=$(<score.csv) 
            csv_string+="${_STR}"
        done
    done 

    IFS=""
    # Compute score
    csv_string=${csv_string//$'\n'/,} # replace \n with ,
    echo ${csv_string} > score_${1}_${2}.csv

    all_string+="${1}|${2}${csv_string}\n"
done
echo -e ${all_string} > all_scores.csv


         



