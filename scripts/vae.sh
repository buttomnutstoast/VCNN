th main.lua \
-epochSize 5000 \
-batchSize 256 \
-nGPU 1 \
-nDonkeys 1 \
-nEpochs 20 \
-netType vae \
-dataset mscoco_decouple \
-data data/MSCOCO \
-train -test
