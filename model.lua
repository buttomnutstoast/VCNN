--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'optim'
if opt.backend == 'cudnn' then require 'cudnn' end

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
local config = opt.netType
netObj = paths.dofile('models/' .. config .. '.lua')
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU, netObj) -- defined in util.lua
else
   print('=> Creating model from file: models/' .. config .. '.lua')
   model = netObj.createModel(opt.nGPU) -- for the model creation code, check the models/ folder
end

-- 2. Create Criterion
if opt.iterSize > 1 then
    origCriterion = netObj.createCriterion()
    criterion = nn.MultiCriterion()
    criterion:add(origCriterion, 1/opt.iterSize)
else
    criterion = netObj.createCriterion()
    origCriterion = criterion
end

print('=> Model')
print(model)

print('=> Criterion')
print(origCriterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
