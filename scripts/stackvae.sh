th main.lua \
-epochSize 5000 \
-batchSize 256 \
-nGPU 1 \
-nDonkeys 1 \
-nEpochs 20 \
-netType stackvae \
-dataset mscoco_decouple \
-data data/MSCOCO \
-train -test
