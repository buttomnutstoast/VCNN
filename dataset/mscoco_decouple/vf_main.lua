require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------------------
--[[
   Section 0: Load command line options
--]]
require 'paths'
local opts = {}

function opts.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 feature extraction script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache',      'checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-cache_file', 'vfProbCache.t7', 'name of file saving visual features and mil_prob')
    cmd:option('-data',       'data/MSCOCO', 'dataset folder')
    cmd:option('-manualSeed',  2,       'Manually set RNG seed')
    cmd:option('-GPU',         1,       'Default preferred GPU')
    cmd:option('-nGPU',        1,       'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | nn')
    ------------- Data options ------------------------
    cmd:option('-imageSize',   256,     'Smallest side of the resized image')
    cmd:option('-imageCrop',   224,     'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',    1000,    'number of classes in the dataset')
    ------------- Model options --------------------
    cmd:option('-nEpochs',     1,       'Number of total epochs to run')
    cmd:option('-batchSize',   64,     'mini-batch size (1 = pure stochastic)')
    cmd:option('-milvc_vgg',  'checkpoint/milvc/milvc_vgg.t7', 'Path to the trained milvc vgg model')
    cmd:option('-vocabs',     'vocabs/vocab_words.txt','Path to the file of 1000 common vocabs in MSCOCO')
    cmd:text()

    ------------ Options from sepcified network -------------
    local netType = ''
    local backend = 'cudnn'
    for i=1, #arg do
        if arg[i] == '-backend' then
            backend = arg[i+1]
        end
    end

    local opt = cmd:parse(arg or {})
    -- append dataset to cache name
    opt.cache = path.join(opt.cache, 'mscoco_decouple')
    -- fix number of donkey to 1 (important), otherwise, order of data will
    -- be polluted!
    opt.nDonkeys = 1
    return opt
end

opt = opts.parse(arg)
nClasses = opt.nClasses
if opt.backend == 'cudnn' then require 'cudnn' end

-- a cache file of the visual features of images (if doesnt exist, will be created)
local cache = paths.concat(opt.cache, opt.cache_file)
if paths.filep(cache) then
    print('Cache exists, skip loading milvc models and feature/mil_probability extraction...')
    return
end

dofile('util.lua') -- load multi-gpu methods

--------------------------------------------------------------------------------
--[[
   Section 1: Construct Visual filter
--]]
netObj = {}
netObj.vf_dim = 4096
function netObj.packages()
    if not nn.ConstantAdd then dofile('layers/ConstantAdd.lua') end
    if not nn.AddSingletonDim then dofile('layers/AddSingletonDim.lua') end
    if not nn.RemoveLastSingleton then dofile('layers/RemoveLastSingleton.lua') end
end

function netObj.createModel(nGPU)
    netObj.packages()

    local milvc_vgg = removeParallelTb(opt.milvc_vgg)
    local cnn = milvc_vgg:get(1)
    local nor = milvc_vgg:get(2)

    -- decouple fc8 from cnn
    local fc8 = nn.Sequential():add(cnn:get(38)):add(cnn:get(39))
    cnn:remove(39)
    cnn:remove(38)

    -- split two outpus {cnn_output, nor_output}
    local split_output = nn.ConcatTable()
    local cnn_output = nn.Sequential():add(nn.View(-1, netObj.vf_dim)) -- fc7 feature dimension should be the same
    local nor_output = nn.Sequential():add(fc8):add(nor) -- connect fc8 with nor classifier
    split_output:add(cnn_output):add(nor_output)

    collectgarbage()

    local cnn_nor_decouple = nn.Sequential():add(cnn):add(split_output)
    local dpt = makeDataParallel(cnn_nor_decouple, nGPU, netObj)
    local model = nn.Sequential():add(dpt):cuda()

    return model
end

function netObj.outputInit()
    local info = {}
    info[#info+1] = newInfoEntry('vf',0,0, true)
    info[#info+1] = newInfoEntry('mil_prob',0,0, true)
    return info
end

function netObj.outputWrite(info, outputs)
    local vf = outputs[1]:float()
    local mil_prob = outputs[2]:float()

    info[1].value = vf
    info[2].value = mil_prob
end

model = netObj.createModel(opt.nGPU)

print('=> Model')
print(model)

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.cache)
os.execute('mkdir -p ' .. opt.cache)

--------------------------------------------------------------------------------
--[[
   Section 2: Create multiple threads for loading images,
--]]

-- Perform visual feature extraction
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local ndonkeys = opt.nDonkeys -- Open all GPUs
do -- start K datathreads (donkeys)
    if ndonkeys > 0 then
        local options = opt -- make an upvalue to serialize over to donkey threads
        donkeys = Threads(
            ndonkeys,
            function()
                require 'torch'
            end,
            function(idx)
                opt = options -- pass to all donkeys via upvalue
                tid = idx
                local seed = opt.manualSeed + idx
                torch.manualSeed(seed)
                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                paths.dofile('vf_donkey.lua')
            end
      );
    else -- single threaded data loading. useful for debugging
        paths.dofile('vf_donkey.lua')
        donkeys = {}
        function donkeys:addjob(f1, f2) f2(f1()) end
        function donkeys:synchronize() end
    end
end

-- Retrieve number of data
nTrain, nTest, nEval = 0, 0, 0
donkeys:addjob(function() return trainLoader:size() end, function(c) nTrain = c end)
donkeys:synchronize()
donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
donkeys:synchronize()
donkeys:addjob(function() return evalLoader:size() end, function(c) nEval = c end)
donkeys:synchronize()

-- Extract visual features and mil-probability
paths.dofile('vf_extract.lua')
local vf_milprob = {}
vf_milprob.trainCache = vf_extract('train')
vf_milprob.evalCache = vf_extract('eval')
vf_milprob.testCache = vf_extract('test')

torch.save(cache, vf_milprob)
