pip install -r requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U transformers
pip install -U datasets
sudo mount -o size=64097152k -o nr_inodes=1000000 -o noatime, nodiratime -o remount /dev/shm
