#!/usr/bin/env bash

MAINTASK=segmentation #it could be: segmentation, parsing, autoparsing
SUBTASK=edu #it could be: edu, span, ns, rela, full, autoedu, autospan, autons, autorela


if [ "$MAINTASK" != "autoparsing" ]; then

    #Constructing task-specific Linearized tree for candidate sentences, while do tokenization on them with tokenizer of MPNet.
    for SPLIT in train valid test; do \
        python scripts/binary_with_task.py --text-file data/$MAINTASK/$SUBTASK/$SPLIT/output_raw.txt \
            --pretrained-mpnet-path mpnet.base/ --task $SUBTASK
    done

    #Constructing task-specific Linearized tree for gold sentences, while do tokenization on them with tokenizer of MPNet.
    for SPLIT in train valid; do \
        python scripts/encode.py \
            --binary-method $SUBTASK \
            --inputs data/$MAINTASK/$SUBTASK/$SPLIT/reference_raw.txt \
            --outputs data/$MAINTASK/$SUBTASK/$SPLIT/reference.bpe \
            --keep-empty \
            --workers 60; \
    done
    
    #Constructing fairseq data
    fairseq-preprocess \
        --only-source \
        --srcdict mpnet_by_microsoft/MPNet/dict.txt \
        --trainpref data/$MAINTASK/$SUBTASK/train/reference.bpe \
        --validpref data/$MAINTASK/$SUBTASK/valid/reference.bpe \
        --destdir data-bin/$MAINTASK/$SUBTASK/ \
        --workers 60


else 

    #Constructing task-specific Linearized tree for candidate sentences, while do tokenization on them with tokenizer of MPNet.
    #For autoparsing, we directly use trained parsers to do evaluation.
    for SPLIT in valid test; do \
        python scripts/binary_with_task.py --text-file data/$MAINTASK/$SUBTASK/$SPLIT/output_raw.txt \
            --pretrained-mpnet-path mpnet.base/ --task $SUBTASK
    done

fi