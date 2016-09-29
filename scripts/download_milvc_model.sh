#!/usr/bin/bash

mkdir checkpoint
mkdir checkpoint/milvc

# get prototxt
wget https://raw.githubusercontent.com/s-gupta/visual-concepts/master/output/vgg/mil_finetune.prototxt.deploy
mv mil_finetune.prototxt.deploy checkpoint/milvc

wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/trained-coco.v2.tgz && tar -xf trained-coco.v2.tgz
mv code/output/vgg/* checkpoint/milvc

rm trained-coco.v2.tgz
rm -rf code
