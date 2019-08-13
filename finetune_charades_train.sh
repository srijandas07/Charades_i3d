#!/usr/bin/env bash
#
export PATH=/home/sdas/anaconda2/bin:$PATH
export PATH=/usr/sbin/:$PATH
module load cuda/8.0 cudnn/5.1-cuda-8.0 opencv/3.4.1
#
mkdir -p /data/stars/user/rdai/charades/charades_SSD
sudo mountimg /data/stars/share/charades/ssd_images.squashfs /data/stars/user/rdai/charades/charades_SSD
#unsquashfs -d /dev/shm/full_body_charades /data/stars/share/charades/ssd_images.squashfs

mkdir -p weights_$2
python /data/stars/user/rdai/charades/I3D_charades/i3d_train_charades.py $1 $2
