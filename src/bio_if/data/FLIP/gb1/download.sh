# download a list of files with wget
# Usage: bash download.sh

# list of files to download
files=(
    "http://data.bioembeddings.com/public/FLIP/fasta/gb1/one_vs_rest.fasta"
    "http://data.bioembeddings.com/public/FLIP/fasta/gb1/two_vs_rest.fasta"
    "http://data.bioembeddings.com/public/FLIP/fasta/gb1/three_vs_rest.fasta"
    "http://data.bioembeddings.com/public/FLIP/fasta/gb1/low_vs_high.fasta"
    "http://data.bioembeddings.com/public/FLIP/fasta/gb1/sampled.fasta"
)

# download files
for file in "${files[@]}"; do
    curl -O $file
done