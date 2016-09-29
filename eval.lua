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

evalLogger = optim.Logger(paths.concat(opt.save, 'eval.log'))

local evalDataIterator = function()
   evalLoader:reset()
   return function() return evalLoader:get_batch(false) end
end

local batchNumber
local timer = torch.Timer()
local savevals, vals

function eval()
   collectgarbage()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   savevals = netObj.evalOutputInit()
   vals = netObj.evalOutputInit()
   for i=1,#savevals do
      if savevals[i].store then
         savevals[i].value = {}
      end
   end
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i=1,torch.ceil(nEval/opt.batchSize) do -- nEval is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      if i == torch.ceil(nEval/opt.batchSize) then indexEnd = nEval end -- no data will be ignored
      local currentBatchNumber = i
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = evalLoader:getInputs(indexStart, indexEnd)
            return inputs, labels, indexStart, indexEnd, currentBatchNumber
         end,
         -- callback that is run in the main thread once the work is done
         evalBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   -- print information
   local strout = ('Epoch: [%d][EVALUATING SUMMARY] Total Time(s): %.2f'):format(epoch, timer:time().real)
   local substrout = ''
   local loggerList = {}
   local joinNN = nn.JoinTable(1)
   for k=1,#savevals do
         substrout = 'avg.'..savevals[k].name..':%.5f'
         loggerList['avg.'..savevals[k].name..' (eval set)'] = savevals[k].value/savevals[k].N
         strout = strout.. ' '..substrout:format(savevals[k].value/savevals[k].N)
   end
   print(strout)
   print('\n')

   evalLogger:add(loggerList)

end -- of eval()
-----------------------------------------------------------------------------
local inputsGPUTable = {}
local labelsGPUTable = {}
local inputs = nil
local labels = nil

function evalBatch(inputsCPU, labelsCPU, sInd, eInd, currentBatchNumber)
   put2GPU(inputsCPU, inputsGPUTable)
   put2GPU(labelsCPU, labelsGPUTable)
   if #inputsGPUTable == 1 then
      inputs = inputsGPUTable[1]
      inputsCPU = inputsCPU[1]
   else
      inputs = inputsGPUTable
   end
   if #labelsGPUTable == 1 then
      labels = labelsGPUTable[1]
      labelsCPU = labelsCPU[1]
   else
      labels = labelsGPUTable
   end

   local outputs, err
   if netObj.ftest then
      outputs, err = netObj.ftest(inputs, labels)
   else
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
   end

   cutorch.synchronize()

   netObj.evalOutput(vals, outputs, labelsCPU, err)
   for k = 1,#vals do
      savevals[k].value = savevals[k].value + vals[k].value*vals[k].N
      savevals[k].N     = savevals[k].N     + vals[k].N
   end

   batchNumber = batchNumber + (eInd-sInd+1)
   if batchNumber % 1024 == 0 then
      print(('Epoch: Evaluating [%d][%d/%d]'):format(epoch, batchNumber, nEval))
   end

   inputs = nil
   labels = nil
end
