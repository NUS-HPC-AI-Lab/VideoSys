# Download the webvid dataset
mkdir datasets
wget https://huggingface.co/datasets/TempoFunk/webvid-10M/resolve/main/data/val/partitions/0000.csv?download=true -O datasets/webvid.csv

# Download video files
python download_webvid.py
