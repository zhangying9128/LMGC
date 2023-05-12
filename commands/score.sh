#!/usr/bin/env bash
# /raid/zhang/RST/mpnet-RST/mpnet.base/mpnet.pt

MAIN_TASK=segmentation #it can be: segmentation, parsing
SUBTASK=edu #it can be: edu, span, ns, rela, full, autoedu, autospan, autons, autorela
GEN_SUBSET=test
LABEL_EMBEDDING=normal #it can be: normal, enhance, concat
DATA_DIR=data-bin/$MAIN_TASK/$SUBTASK/
MODEL_DIR=checkpoints/$MAIN_TASK/$SUBTASK/discriminative_20_solvedata_lr0.00009_1/
REFERENCE_DIR=data/$MAIN_TASK/$SUBTASK/$GEN_SUBSET/

CUDA_VISIBLE_DEVICES=0 fairseq-score $DATA_DIR \
    --path $MODEL_DIR/checkpoint_best.pt \
    --discriminative-type $SUBTASK \
    --reference-dir $REFERENCE_DIR \
    --reference gold_edu.txt \
    --prediction pred_${SUBTASK}_binary.txt \
    --label-embedding $LABEL_EMBEDDING \
    --score-file $MODEL_DIR/${GEN_SUBSET}_${MAIN_TASK}_${SUBTASK}_scores.txt \
    --task masked_permutation_lm --criterion masked_permutation_cross_entropy \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6
