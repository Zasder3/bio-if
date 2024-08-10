# model_name = "facebook/esm2_t36_3B_UR50D"
# model_name = "facebook/esm2_t33_650M_UR50D"
# model_name = "facebook/esm2_t30_150M_UR50D"
# model_name = "facebook/esm2_t12_35M_UR50D"
# model_name = "facebook/esm2_t6_8M_UR50D"

# Array of study names
study_names=(
    # "GCN4_YEAST_Staller_2018"
    # "Q8WTC7_9CNID_Somermeyer_2022"
    "CAPSD_AAV2S_Sinai_2021"
    # "HIS7_YEAST_Pokusaeva_2019"
    # "D7PM05_CLYGR_Somermeyer_2022"
    # "GFP_AEQVI_Sarkisyan_2016"
    # "PHOT_CHLRE_Chen_2023"
    # "Q6WV13_9MAXI_Somermeyer_2022"
    # "F7YBW7_MESOW_Ding_2023"
    # "F7YBW8_MESOW_Aakre_2015"
)

# Loop through each study name and run the commands
for study_name in "${study_names[@]}"
do
    # CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc-per-node=4 all-vs-wt.py --per_device_batch_size=8 --model_name="facebook/esm2_t6_8M_UR50D" --study_name="$study_name"
    # CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc-per-node=4 all-vs-wt.py --per_device_batch_size=8 --model_name="facebook/esm2_t12_35M_UR50D" --study_name="$study_name"
    # CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc-per-node=4 all-vs-wt.py --per_device_batch_size=8 --model_name="facebook/esm2_t30_150M_UR50D" --study_name="$study_name"
    # CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc-per-node=4 all-vs-wt.py --per_device_batch_size=8 --model_name="facebook/esm2_t33_650M_UR50D" --study_name="$study_name"
    # CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc-per-node=4 all-vs-wt.py --per_device_batch_size=4 --model_name="facebook/esm2_t36_3B_UR50D" --study_name="$study_name"
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc-per-node=4 all-vs-wt.py --per_device_batch_size=1 --model_name="facebook/esm2_t36_3B_UR50D" --study_name="$study_name"
done
