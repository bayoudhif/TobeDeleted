cd biomedclip_finetuning/open_clip/src

## You can change the following parameters like the GPU devices, batch size, training data, epochs, and DHN-NCE loss parameters.

CUDA_VISIBLE_DEVICES=0 python3 -m open_clip_train.main \
    --batch-size 16 \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data data/medpix_dataset/medpix_dataset.csv \
    --csv-img-key filename \
    --csv-caption-key Caption \
    --lr=1e-5 \
    --wd=0.1 \
    --warmup 0 \
    --epochs=150 \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --pretrained weights/your_weights.bin \
    --contrastive-loss \
    --temperature-contrastive 0.07 \
    --lock-image \
    --lock-image-freeze-bn-stats