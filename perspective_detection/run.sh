#!/bin/bash

# TEST_OR_DEV="dev"
# DATA="hc/sparse_dev_hc.tsv"
TEST_OR_DEV="test"
DATA="subtask1.test_unique.tsv"
ROOT="/scratch/cluster/hungting/projects/Multi_Answer/Data/subtask_1"

# rerun - gemma (refined), llama70b

# INSTRUCTION="nli_${CHAT_STR}cot"
# INSTRUCTION="nli${CHAT_STR}"

# gemma, mistral, zephyr
# mistralai/Mistral-${SIZE}-Instruct-v0.2
# HuggingFaceH4/zephyr-${SIZE}-beta
# google/gemma-${SIZE}-it
# llama-2-${SIZE}-chat
# gpt4

# for MODEL_SHORT in "llama" 
for MODEL_SHORT in "t5"
do
    if [ ${MODEL_SHORT} == "llama" ]
    then
        SIZES="70b"
        # SIZES="7b 13b 70b"
    elif [ ${MODEL_SHORT} == "gpt4" ]
    then
        SIZES="0"
        NUM_NODE=1
    else
        SIZES="7b"
        NUM_NODE=1
    fi

    for SIZE in ${SIZES}
    do
        echo ${SIZE}
        if [ ${MODEL_SHORT} == "llama" ]
        then
            MODEL_NAME="llama-2-${SIZE}-chat"
            if [ ${SIZE} == "7b" ]
            then
                NUM_NODE=1
            elif [ ${SIZE} == "13b" ]
            then
                NUM_NODE=2
            elif [ ${SIZE} == "70b" ]
            then
                NUM_NODE=8
            else
                echo "Invalid size"
                exit 1
            fi
            CHAT="chat" # chat or no-chat
        elif [ ${MODEL_SHORT} == "gpt4" ]
        then
            MODEL_NAME="gpt4"
            CHAT="chat" # chat or no-chat
        elif [ ${MODEL_SHORT} == "gemma" ]
        then
            MODEL_NAME="google/gemma-${SIZE}-it"
            CHAT="no-chat" # chat or no-chat
        elif [ ${MODEL_SHORT} == "mistral" ]
        then
            MODEL_NAME="mistralai/Mistral-${SIZE}-Instruct-v0.2"
            # MODEL_NAME="/scratch/cluster/hungting/projects/Multi_Answer/Subtask_1/saved_models/Mistral_test_ep2_lr1e-4_wc0.001_wr0.1_schlinear"
            # CHAT="no-chat" # chat or no-chat
            CHAT="chat" # chat or no-chat
        elif [ ${MODEL_SHORT} == "zephyr" ]
        then
            MODEL_NAME="HuggingFaceH4/${MODEL_SHORT}-${SIZE}-beta"
            CHAT="chat" # chat or no-chat
        elif [ ${MODEL_SHORT} == "t5" ]
        then
            MODEL_NAME="google/t5_xxl_true_nli_mixture"
            CHAT="chat" # chat or no-chat
        else
            echo "Invalid model"
            exit 1
        fi


        if [ ${CHAT} == "chat" ]
        then
        CHAT_STR="_chat"
        else
        CHAT_STR=""
        fi
        # for INSTRUCTION in "nli${CHAT_STR}" "nli${CHAT_STR}_support" "nli${CHAT_STR}_1shot" "nli${CHAT_STR}_2shot" "nli${CHAT_STR}_cot" "nli${CHAT_STR}_refined" "nli${CHAT_STR}_refined_1shot" "nli${CHAT_STR}_refined_2shot" "nli${CHAT_STR}_refined_cot"
        for INSTRUCTION in "nli${CHAT_STR}_refined" 
        do
            if [ ${MODEL_SHORT} == "gpt4" ]
            then
                OUTFILE="predictions/${TEST_OR_DEV}/${MODEL_SHORT}/pred_${MODEL_SHORT}–${INSTRUCTION}_unique.tsv"
            else
                OUTFILE="predictions/${TEST_OR_DEV}/${MODEL_SHORT}-${SIZE}/pred_${MODEL_SHORT}_${SIZE}–${INSTRUCTION}_unique.tsv"
            fi
            echo $MODEL_NAME
            echo "${ROOT}/${DATA}"
                    torchrun --nproc_per_node ${NUM_NODE} --master-port 29600  subtask1.py \
                    --data ${ROOT}/${DATA}  \
                    --output_file ${OUTFILE} \
                    --instructions instructions/${CHAT}/instructions_${INSTRUCTION}.txt \
                    --model ${MODEL_NAME}  \
                    --model_type ${MODEL_SHORT} \
                    --write_results --write_scores
        done

    done
done
            # torchrun --nproc_per_node ${NUM_NODE} subtask1.py \


         



