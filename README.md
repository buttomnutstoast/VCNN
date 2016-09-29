# VCNN

This code implements our paper **Variational Convolutional Networks for Human-Centric Annotations**. We integrate a CNN with a varitaional auto-encoder (VAE) to tackle human-centric annotations.

## Requirement

  1. [torch](http://torch.ch/)
  2. CUDA (only test on CUDA 7.5)
  3. cudnn (only test on cudnn v4)

After installing torch, we need few more packages

```
$ luarocks install nn
$ luarocks install cutorch
$ luarocks install cunn
$ luarocks install cudnn
$ luarocks install nngraph
$ luarocks install optim
$ luarocks install image
$ luarocks install hdf5
$ luarocks install tds
$ luarocks install json
$ luarocks install dkjson
$ luarocks install loadcaffe
```

The CNN we used is first fine-tuned over MsCOCO for detecting visual concepts (see [this paper](https://people.eecs.berkeley.edu/~sgupta/pdf/captions.pdf)). Download their [pre-trained vgg model](https://github.com/s-gupta/visual-concepts) and save it do checkpoint/milvc

```
$ bash scripts/download_milvc_model.sh
```

Load the caffemodel and transfer it to torch-compatible format (we use cudnn by default, you could rather choose nn by appending "-backend nn" to the following command)

```
$ th utils/milvcvgg_for_vcnn.lua
```

## Training

#### For small images (224x224)

(1) train VAE (you could set learning rate and weight decay by adding "-LR 0.01" and "-weightDecay 5e-4" to the following commands or you can directly edit function trainRule in models/vae.lua)

```
$ th dataset/mscoco_decouple/vf_main.lua
$ bash scripts/vae.sh
```

(2) train stacked-VAE 
first your should modify function arguments in models/stackvae.lua, and set trained vae path (for example, cmd:option('-prev_vae', 'checkpoint/mscoco_decouplt/trained_vae/model_20.t7')) or the code will re-initiate a vae

```
$ bash scripts/stackvae.sh
```

(3) train VCNN (set trained vae path at models/vcnn.lua first)

```
$ bash scripts/vcnn.sh
```

(4) train stacked-VCNN (set trained stacked-vae path at models/stackvcnn.lua first)

```
$ bash scripts/stackvcnn.sh
```

#### For larger images (565x565)

(5) train milvc-stacked-VCNN (set trained stacked-vae path at models/milvc_stackvcnn.lua first)

```
$ bash scripts/milvc_stackvcnn.sh
```

#### Evaluation

We borrow [Saurabh's code](https://github.com/s-gupta/visual-concepts). Choose the output file of testing set at iteration n. And run

```
$ python scripts/model_eval.py --det_file /path/to/testOutput_n.t7
```

## MSCOCO dataset

(1) Download images

```
$ bash scripts/download_mscoco.sh
```

(2) Download [captions](http://mscoco.org/dataset/#download) and transform it to tds (torch-compatible data format)
```
th utils/loadMSCOCO.lua -split train
th utils/loadMSCOCO.lua -split val
```

(3) Download and install the [API](https://github.com/pdollar/coco)

======================================================

## Overview

We base our codes on [the package](https://github.com/soumith/imagenet-multiGPU.torch) written by Soumith. To use his code, we have followed [his licence](https://github.com/soumith/imagenet-multiGPU.torch/blob/master/LICENSE.md). If you want to redistribute or use our codes, please also follow Soumith's License.

This package is extended from codes written by Soumith which provides easy ways to import and export dataset, which is used in training/testing/evaluating deep-net models by torch7.

The package is consist of following parts:

    1. Loading Command Line Options (opts.lua)
    2. Parallel Computations on Multi-GPUs (util.lua)
    3. Deep-net Model Construction (model.lua)
    4. Parallel Data Loading (data.lua)

You should write your own codes in:

    1. models/YOURMODEL.lua
    2. donkey.lua and dataset.lua in dataset/YOURDATASET

**NOTE**: please make sure that

    1. functions including createModel, createCriterion, trainOutputInit,
       trainOutput, testOutputInit, testOutput, evalOutputInit, evalOutput,
       trainRule (or your should manually assing Learning Rate & Weight Decay)
       are implemneted in models/YOURMODEL.lua
    2. functions getInputs, genInput, get and sample are implemented in
       dataset/YOURDATASET/dataset.lua; getInputs and genInput should return
       a table of input-data and a table of ground-truth labels
    3. functions sampleHookTrain and sampleHoodTest are implemented in
       dataset/YOURDATASET/donkey.lua which explicitly load and process input
       data

There are global variables:

    1. model (your own models)
    2. criterion (your own loss-criterion)
    3. model (which defines how to create and optimize models)
    4. donkeys (parallel threads to load data)
    5. epoch (current epoch in training/testing/evaluating)
