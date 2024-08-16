# loop through the folder anlalysis-seq for each FASTA
# then run the analyze-sequence.py program for each FASTA file
for file in analysis-seq/*.fasta
do
    if [ "$file" == "analysis-seq/combined.fasta" ]; then
        continue
    fi
    data="training-seq/$(basename "$file" .fasta).csv"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 analyze-sequence.py --sequence_of_interest $file --training_data $data
done