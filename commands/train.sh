#!/usr/bin/env bash
MAIN_TASK=parsing # it could be: segmentation, parsing
SUB_TASK=rela #it could be: edu, span, ns, rela, full

if [ "$SUB_TASK" == "edu" ]; then
    TOTAL_UPDATES=17550
    WARMUP_UPDATES=1404
elif [ "$SUB_TASK" == "span" ]; then
    TOTAL_UPDATES=10800
    WARMUP_UPDATES=864
elif [ "$SUB_TASK" == "ns" ]; then
    TOTAL_UPDATES=10800
    WARMUP_UPDATES=864
elif [ "$SUB_TASK" == "rela" ]; then
    TOTAL_UPDATES=14940
    WARMUP_UPDATES=1196
elif [ "$SUB_TASK" == "full" ]; then
    TOTAL_UPDATES=14970
    WARMUP_UPDATES=1198
else
    echo "$SUBTASK error"
fi

PEAK_LR=0.00009          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_TOKENS=8192        # Number of sequences per batch (batch size)
UPDATE_FREQ=1          # Increase the batch size 1x
SEED=1
LABEL_EMBEDDING=enhance
CANDIDATE_SIZE=20
SAVE_PATH=checkpoints/$MAIN_TASK/$SUB_TASK/${LABEL_EMBEDDING}_${SEED}/
mkdir -p $SAVE_PATH

DATA_DIR=data-bin/$MAIN_TASK/$SUB_TASK/

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file mpnet.base/mpnet.pt \
    --seed $SEED \
    --no-epoch-checkpoints \
    --no-save-optimizer-state \
    --reset-optimizer --reset-dataloader --reset-meters \
    --task masked_permutation_lm --criterion masked_permutation_cross_entropy \
    --arch mpnet_base \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --update-freq $UPDATE_FREQ \
    --sample-break-mode 'eos' \
    --label-embedding  $LABEL_EMBEDDING \
    --discriminative-loss \
    --discriminative-size $CANDIDATE_SIZE \
    --discriminative-type $SUB_TASK \
    --train-cand-path data/$MAIN_TASK/$SUB_TASK/train/ \
    --valid-cand-path data/$MAIN_TASK/$SUB_TASK/valid/ \
    --reference reference_raw.txt \
    --prediction output_${SUB_TASK}_binary.txt \
    --max-epoch 30 \
    --use-relative-positions \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 60 --input-mode 'mpnet' 