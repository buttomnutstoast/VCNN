-- implementation of
-- Learning Visual Classifiers using Human-centric Annotations
--

local NET = {}
function NET.packages()
   if not nn.ConstantAdd then dofile('layers/ConstantAdd.lua') end
   if not nn.AddSingletonDim then dofile('layers/AddSingletonDim.lua') end
end

function NET.createModel(nGPU)
   NET.packages()
   local caffe2torch = require('utils/caffe2torch/caffe2torch')
   existedModel = caffe2torch(opt.prototxt, opt.caffemodel, opt.backend, true)
   local features = makeDataParallel(existedModel, nGPU, NET) -- defined in util.lua

   -------------------------------------------------------

   collectgarbage()

   local nor_path = nn.Sequential()
   nor_path:add(nn.ConstantAdd(1+1e-12,-1))
   nor_path:add(nn.Log())  -- output = log(N x 1000 x h x w)
   nor_path:add(nn.Sum(-1)):add(nn.Sum(-1)) -- output = log(N x 1000)
   nor_path:add(nn.Exp())
   nor_path:add(nn.Threshold(1e-7, 1e-7))
   nor_path:add(nn.ConstantAdd(1, -1))
   nor_path:add(nn.AddSingletonDim()) -- add dimension to N x 1000 x 1

   local max_path = nn.Sequential()
   max_path:add(nn.Max(-1)):add(nn.Max(-1)) -- ouput = N x 1000
   max_path:add(nn.AddSingletonDim()) -- add dimension to N x 1000 x 1

   local nor_and_max_path = nn.ConcatTable()
   nor_and_max_path:add(nor_path):add(max_path)

   local classifier = nn.Sequential()
   classifier:add(nor_and_max_path)
   classifier:add(nn.JoinTable(3))
   classifier:add(nn.Max(3))
   classifier:add(nn.View(-1, nClasses))

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)
   model:cuda()

   return model

end

function NET.createCriterion()
   return nn.BCECriterion()
end

function NET.trainOutputInit()
   local info = {}
   info[#info+1] = newInfoEntry('loss',0,0) -- last number  = batch_size. must >0
   info[#info+1] = newInfoEntry('map',0,0)
   return info
end

function NET.trainOutput(info, outputs, labels, err)
   local batch_size = outputs:size(1)
   local outputsCPU = outputs:float()

   info[1].value   = err*opt.iterSize
   info[1].N       = batch_size

   info[2].value   = meanAvgPrec(outputsCPU, labels, 2, opt.threshold)
   info[2].N       = batch_size
end

function NET.testOutputInit()
   local info = {}
   info[#info+1] = newInfoEntry('mil_prob',0,0, true)
   info[#info+1] = newInfoEntry('map',0,0)
   return info
end

function NET.testOutput(info, outputs, labels, err)
   local batch_size = outputs:size(1)
   local outputsCPU = outputs:float()
   info[1].value = outputsCPU
   info[2].value = topK(outputsCPU, labels, 1)
   info[2].N     = batch_size
end

function NET.evalOutputInit()
   local info = {}
   info[#info+1] = newInfoEntry('map',0,0)
   return info
end

function NET.evalOutput(info, outputs, labels, err)
   local batch_size = outputs:size(1)
   local outputsCPU = outputs:float()
   info[1].value = topK(outputsCPU, labels, 1)
   info[1].N     = batch_size
end

function NET.gradProcessing(createdMode, modelPa, modelGradPa, currentEpoch)
   -- processing!!!
end

function NET.arguments(cmd)
   cmd:option('-vocabs', 'vocabs/vocab_words.txt','Path to the file of 1000 common vocabs in MSCOCO')
   cmd:option('-rMean', '123.68', 'mean pixel value of channel R')
   cmd:option('-gMean', '116.779', 'mean pixel value of channel G')
   cmd:option('-bMean', '103.939', 'mean pixel value of channel B')
   cmd:option('-prototxt', 'checkpoint/milvc/mil_finetune.prototxt.deploy', 'path of defined prototxt')
   cmd:option('-caffemodel', 'checkpoint/milvc/vgg_16_full_conv.caffemodel', 'path of vgg model')
   cmd:option('-nnType', 'linear', 'specify different output format, vae | vae_2 | linear')
   cmd:option('-threshold', 0.1, 'threshold for prediction')
end

function NET.trainRule(currentEpoch)
   local minVal = 5 -- 10^-4
   local maxVal = 3 -- 10^-1
   local ExpectedTotalEpoch = 5
   return {LR= 10^-((currentEpoch-1)*(minVal-maxVal)/(ExpectedTotalEpoch-1)+1),
           WD= 5e-4}
end

return NET
