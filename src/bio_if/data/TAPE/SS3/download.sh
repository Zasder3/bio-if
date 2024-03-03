# download a list of files with wget
# Usage: bash download.sh

# list of files to download
files=(
    http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/secondary_structure.tar.gz    
)

# download files
for file in "${files[@]}"; do
    curl -O $file
done