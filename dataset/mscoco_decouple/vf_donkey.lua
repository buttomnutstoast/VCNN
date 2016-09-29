--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
require 'cutorch'
paths.dofile('vf_dataset.lua')
local currentDir = paths.dirname(paths.thisfile())

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.imageCrop, opt.imageCrop}

--------------------------------------------------------------------------------
--[[
   Section 0: Create image loader functions each for training ane testing,
   the function in training upscale the shorter side to loadSize, however,
   the one in testing upscale the longer side.
--]]
local function loadImage(path)
    local inputRGB = image.load(path, 3, 'float')
    -- find the smaller side, and resize it to loadSize (while keeping aspect ratio)
    local sideLen = math.min(inputRGB:size(3), inputRGB:size(2))
    assert(sideLen > 0, 'side length of input image should be larger than 0')
    local ratio = {-1, inputRGB:size(2)/sideLen, inputRGB:size(3)/sideLen}
    inputRGB = image.scale(inputRGB, loadSize[3]*ratio[3], loadSize[2]*ratio[2])
    local inputBGR = torch.Tensor(inputRGB:size()):fill(0)
    for ch=1,3 do
        inputBGR[{{4-ch}, {}, {}}] = inputRGB[{{ch}, {}, {}}]
    end
    return inputBGR
end

-- channel-wise mean and rescale. Calculate or load them from disk later in the script.
local mean, rescale
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
    collectgarbage()
    local input = loadImage(path)
    local oH = sampleSize[2]
    local oW = sampleSize[3]
    local iW = input:size(3)
    local iH = input:size(2)
    local w1 = math.ceil((iW-oW)/2)
    local h1 = math.ceil((iH-oH)/2)
    local crop_input = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
    -- mean/rescale
    for i=1,3 do -- channels
        if rescale then crop_input[{{i},{},{}}]:mul(rescale[i]) end
        if mean then crop_input[{{i},{},{}}]:add(-mean[i]) end
    end
    -- padding
    local out = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3]):fill(0)
    out[{{}, {1, crop_input:size(2)}, {1, crop_input:size(3)}}] = crop_input

    return out
end

print('Creating training metadata')
trainLoader = dataLoader{
    path = paths.concat(opt.data),
    vocPath = paths.concat(currentDir, opt.vocabs),
    imageIdPath = paths.concat(currentDir, 'splits', 'train.ids'),
    protocol = 'train',
    split = 100,
    }
trainLoader.imgHook = trainHook
collectgarbage()

-- do some sanity checks on trainLoader
do
    local nClasses = #trainLoader.classes
    assert(nClasses == 1000, "class logic has error")
end

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
local testHook = function(self, path)
    collectgarbage()
    local input = loadImage(path)
    local oH = sampleSize[2]
    local oW = sampleSize[3]
    local iW = input:size(3)
    local iH = input:size(2)
    local w1 = math.ceil((iW-oW)/2)
    local h1 = math.ceil((iH-oH)/2)
    local crop_input = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
    -- mean/rescale
    for i=1,3 do -- channels
        if rescale then crop_input[{{i},{},{}}]:mul(rescale[i]) end
        if mean then crop_input[{{i},{},{}}]:add(-mean[i]) end
    end
    -- padding
    local out = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3]):fill(0)
    out[{{}, {1, crop_input:size(2)}, {1, crop_input:size(3)}}] = crop_input

    return out
end

print('Creating test metadata')
testLoader = dataLoader{
    path = paths.concat(opt.data),
    vocPath = paths.concat(currentDir, opt.vocabs),
    imageIdPath = paths.concat(currentDir, 'splits', 'test.ids'),
    protocol = 'val',
    split = 0,
    }
testLoader.imgHook = testHook
collectgarbage()

-- End of test loader section
--------------------------------------------------------------------------------
--[[
   Section 3: Create a eval data loader (evalLoader),
   which can iterate over the eval set and returns an image's
--]]

-- evalLoader adopts same function as testLoader to load the image
print('Creating eval metadata')
evalLoader = dataLoader{
    path = paths.concat(opt.data),
    vocPath = paths.concat(currentDir, opt.vocabs),
    imageIdPath = paths.concat(currentDir, 'splits', 'eval.ids'),
    protocol = 'val',
    split = 0,
    }
evalLoader.imgHook = testHook
collectgarbage()

-- End of test loader section
-- Estimate the per-channel mean/rescale (so that the loaders can normalize appropriately)
mean = {103.939, 116.779, 123.68}
rescale = {255, 255, 255} -- rescale image data inconsistent with caffe data layer
