#!/usr/bin/bash

mkdir data && mkdir data/MSCOCO
cd data/MSCOCO

mkdir images
cd images

wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip

cd ../../..
