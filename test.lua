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

local hdf5 = require 'hdf5'
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end

local batchNumber
local timer = torch.Timer()
local savevals, vals

function test()
   collectgarbage()
   print('==> doing epoch on testing data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   savevals = netObj.testOutputInit()
   vals = netObj.testOutputInit()
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
   for i=1,torch.ceil(nTest/opt.batchSize) do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      if i == torch.ceil(nTest/opt.batchSize) then indexEnd = nTest end -- no data will be ignored
      local currentBatchNumber = i
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:getInputs(indexStart, indexEnd)
            return inputs, labels, indexStart, indexEnd, currentBatchNumber
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   local dbName = paths.concat(opt.save, 'testOutput_' .. epoch .. '.h5')
   local predH5DB = hdf5.open(dbName, 'w')
   -- print information
   local strout = ('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f'):format(epoch, timer:time().real)
   local substrout = ''
   local loggerList = {}
   local joinNN = nn.JoinTable(1)
   for k=1,#savevals do
      if savevals[k].store then
         predH5DB:write(savevals[k].name, joinNN:forward(savevals[k].value))
         substrout = savevals[k].name..':(saved to disk)'
         loggerList[savevals[k].name..' (test set)'] = '(saved to disk)'
         strout = strout.. ' '..substrout
      else
         substrout = 'avg.'..savevals[k].name..':%.5f'
         loggerList['avg.'..savevals[k].name..' (test set)'] = savevals[k].value/savevals[k].N
         strout = strout.. ' '..substrout:format(savevals[k].value/savevals[k].N)
      end
   end
   predH5DB:close()
   print(strout)
   print('\n')

   testLogger:add(loggerList)

end -- of test()
-----------------------------------------------------------------------------

local inputsGPUTable = {}
local labelsGPUTable = {}
local inputs = nil
local labels = nil

function testBatch(inputsCPU, labelsCPU, sInd, eInd, currentBatchNumber)


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

   netObj.testOutput(vals, outputs, labelsCPU, err)
   for k = 1,#vals do
      if savevals[k].store then
         savevals[k].value[currentBatchNumber] = vals[k].value
      else
         savevals[k].value = savevals[k].value + vals[k].value*vals[k].N
         savevals[k].N     = savevals[k].N     + vals[k].N
      end
   end

   batchNumber = batchNumber + (eInd-sInd+1)
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end

   inputs = nil
   labels = nil
end
