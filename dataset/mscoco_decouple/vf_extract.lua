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

local batchNumber
local timer = torch.Timer()
local savevals, vals
local nData

function vf_extract(loaderType)
   collectgarbage()
   print('==> doing epoch on input data:')
   print("==> online epoch # " .. 1)

   batchNumber = 0
   savevals = netObj.outputInit()
   vals = netObj.outputInit()

   if loaderType == 'train' then
      nData = nTrain
   elseif loaderType == 'test' then
      nData = nTest
   elseif loaderType == 'eval' then
      nData = nEval
   end

   for i=1,#savevals do
      if savevals[i].store then
         savevals[i].value = {}
      end
   end
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   for i=1,torch.ceil(nData/opt.batchSize) do
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      if i == torch.ceil(nData/opt.batchSize) then indexEnd = nData end -- no data will be ignored

      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local dataLoader
            if loaderType == 'train' then
               dataLoader = trainLoader
            elseif loaderType == 'test' then
               dataLoader = testLoader
            elseif loaderType == 'eval' then
               dataLoader = evalLoader
            end
            local inputs, labels = dataLoader:getInputs(indexStart, indexEnd)
            return inputs, labels, indexStart, indexEnd
         end,
         -- callback that is run in the main thread once the work is done
         extractBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   -- print information
   local strout = ('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f'):format(1, timer:time().real)
   local substrout = ''
   local joinNN = nn.JoinTable(1)
   local results = {}
   for k=1,#savevals do
      if savevals[k].store then
         local joinval = joinNN:forward(savevals[k].value)
         results[#results+1] = joinval:clone() -- clone() is crucial!
         substrout = savevals[k].name..':(saved to disk)'
         strout = strout.. ' '..substrout
      end
   end
   print(strout)
   print('\n')
   return results

end -- of test()
-----------------------------------------------------------------------------

local inputsGPUTable = {}
local labelsGPUTable = {}
local inputs = nil
local labels = nil

function extractBatch(inputsCPU, labelsCPU, sInd, eInd)

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

   local outputs = model:forward(inputs)
   cutorch.synchronize()

   netObj.outputWrite(vals, outputs)
   for k = 1,#vals do
      savevals[k].value[#savevals[k].value+1] = vals[k].value
   end

   batchNumber = batchNumber + (eInd-sInd+1)
   if batchNumber % 1024 == 0 then
      print(('Epoch: Extracting [%d][%d/%d]'):format(1, batchNumber, nData))
   end

   inputs = nil
   labels = nil
end
