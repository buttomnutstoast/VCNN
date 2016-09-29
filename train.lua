--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'os'
dofile('utils/train_eval_test_func.lua')

--[[
  1. Setup SGD optimization state and learning rate schedule
  2. Create loggers.
  3. train - this function handles the high-level training loop,
             i.e. load data, train model, save model and state to disk
  4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
  learningRate = opt.LR,
  learningRateDecay = 0.0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
      return {LR= opt.LR, WD= opt.weightDecay}
    end
    return netObj.trainRule(epoch)
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local currentvals, vals, tmpvals
local calltimes = 0

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch)

  local params = paramsForEpoch(epoch)
  optimState = {
    learningRate = params.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = params.WD
  }
  batchNumber = 0
  calltimes = 0
  tmpvals = netObj.trainOutputInit()
  currentvals = netObj.trainOutputInit()
  vals = netObj.trainOutputInit()
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:training()

  local tm = torch.Timer()
  for i=1,opt.epochSize do
    for j=1, opt.iterSize do
      local currentEpoch = epoch
      -- queue jobs to data-workers
      donkeys:addjob(
        -- the job callback (runs in data-worker thread)
        function()
          epoch = currentEpoch -- share epoch with threads
          local inputs, labels = trainLoader:genInputs(opt.batchSize)
          return inputs, labels
          end,
        -- the end callback (runs in the main thread)
        trainBatch
      )
    end
  end

  donkeys:synchronize()
  cutorch.synchronize()

  -- print information
  local strout = ('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f'):format(epoch, tm:time().real)
  local substrout = ''
  local loggerList = {}
  for k=1,#tmpvals do
    substrout = 'avg.'..tmpvals[k].name..':%.5f'
    loggerList['avg.'..tmpvals[k].name..' (train set)'] = tmpvals[k].value/tmpvals[k].N
    strout = strout.. ' '..substrout:format(tmpvals[k].value/tmpvals[k].N)
  end
  print(strout)
  print('\n')

  trainLogger:add(loggerList)

  -- save model
  collectgarbage()

  -- recursively renew element
  local function renew(val, ind)
    if ind > #val then return end
    local item = val[ind]
    if torch.type(item) == 'table' then
      renew(item, 1)
    else
      val[ind] = item.new()
    end
    renew(val, ind+1)
  end

  -- clear the intermediate states in the model before saving to disk
  -- this saves lots of disk space
  -- model:clearState() -- There is a bug, need fixed!
  local mm = model:listModules()
  for mmm=1,#mm do
    if mm[mmm].output then
      if torch.isTensor(mm[mmm].output) then
        mm[mmm].output = mm[mmm].output.new()
      elseif type(mm[mmm].output) == 'table' then
        mm[mmm].output = {}
      else
        mm[mmm].output = nil
      end
    end
    if mm[mmm].gradInput then
      if torch.isTensor(mm[mmm].gradInput) then
        mm[mmm].gradInput = mm[mmm].gradInput.new()
      elseif type(mm[mmm].gradInput) == 'table' then
        mm[mmm].gradInput = {}
      else
        mm[mmm].gradInput = nil
      end
    end
  end
  saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
  -- model = loadDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), opt.nGPU, netObj)
end -- of train()
-------------------------------------------------------------------------------------------
local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()
local pa_noflat, gradPa_noflat = model:parameters()

local inputsGPUTable = {}
local labelsGPUTable = {}
local inputs = nil
local labels = nil

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
  -- GPU inputs (preallocate)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  if calltimes==0 then timer:reset() end

  -- transfer over to GPU
  put2GPU(inputsCPU, inputsGPUTable)
  put2GPU(labelsCPU, labelsGPUTable)
  if #inputsGPUTable == 1 and torch.type(inputsGPUTable[1])~= 'table' then
    inputs = inputsGPUTable[1]
    inputsCPU = inputsCPU[1]
  else
    inputs = inputsGPUTable
  end
  if #labelsGPUTable == 1 and torch.type(labelsGPUTable[1])~= 'table' then
    labels = labelsGPUTable[1]
    labelsCPU = labelsCPU[1]
  else
    labels = labelsGPUTable
  end

  local err, outputs
  if calltimes==0 then
    model:zeroGradParameters()
    currentvals = netObj.trainOutputInit()
  end

  calltimes = calltimes + 1
  if netObj.feval then
    outputs, err = netObj.feval(inputs, labels)
  else
    outputs = model:forward(inputs)
    err = criterion:forward(outputs, labels)
    local gradOutputs = criterion:backward(outputs, labels)
    model:backward(inputs, gradOutputs)
  end
  feval = function(x) return err, gradParameters end

  cutorch.synchronize()
  netObj.gradProcessing(model, pa_noflat, gradPa_noflat, epoch)

  if calltimes == opt.iterSize then
    optim.sgd(feval, parameters, optimState)
    -- DataParallelTable's syncParameters
    if model.needsSync then
      model:syncParameters()
    end
  end
  cutorch.synchronize()

  do
    netObj.trainOutput(vals, outputs, labelsCPU, err)
    for k = 1,#vals do
      currentvals[k].value = currentvals[k].value + vals[k].value*vals[k].N
      currentvals[k].N     = currentvals[k].N     + vals[k].N
    end
  end

  if calltimes == opt.iterSize then
    batchNumber = batchNumber + 1

    -- print information
    local strout = ('%s Epoch: [%d][%d/%d]\tRun:%.3fs lr:%.3e Data:%.3fs'):format(
      os.date("%x %X"), epoch, batchNumber, opt.epochSize, timer:time().real,
      optimState.learningRate, dataLoadingTime)
    local substrout = ''
    for k=1,#currentvals do
      substrout = currentvals[k].name..':%.5f'
      strout = strout.. ' '..substrout:format(currentvals[k].value/currentvals[k].N)
      tmpvals[k].value = tmpvals[k].value + currentvals[k].value
      tmpvals[k].N = tmpvals[k].N + currentvals[k].N
    end
    print(strout)

    dataTimer:reset()
    calltimes = 0
  end

  inputs = nil
  labels = nil
end
