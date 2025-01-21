    # --train_data_path experiments/finetuning/datasets/kennouche_2019_uniref100.csv \
python src/bio_if/training/main.py \
    --dms_id AMFR_HUMAN_Tsuboyama_2023_4G3O \
    --train_data_path experiments/finetuning/datasets/dms_ids_search.parquet.gzip \
    --ood_val_data_path experiments/finetuning/datasets/uniref50_random_10k.csv \
    --fitness_dataset_path experiments/dms/studies/AMFR_HUMAN_Tsuboyama_2023_4G3O.csv \
    --fp16 \
    --lr 1e-6 \
    --batch_size 32