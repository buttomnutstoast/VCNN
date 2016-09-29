require 'torch'
require 'cutorch'
require 'cunn'

dofile('utils/caffe2torch/caffe2torch.lua')
dofile('util.lua')

function parse(args)
    local cmd = torch.CmdLine()
    cmd:text('Options:')
    cmd:option('-prototxt',   'checkpoint/milvc/mil_finetune.prototxt.deploy', '/path/to/milvc/prototxt')
    cmd:option('-caffemodel', 'checkpoint/milvc/snapshot_iter_240000.caffemodel', '/path/to/milvc/caffemodel')
    cmd:option('-save',       'checkpoint/milvc/milvc_vgg.t7', '/path/to/saved/milvc/torchmodel')
    cmd:option('-backend',    'cudnn', 'Options cudnn | nn')
    cmd:option('-GPU',         1, 'set primary GPU')

    local opt = cmd:parse(arg or {})
    return opt
end

opt = parse(args)
if opt.backend == 'cudnn' then require 'cudnn' end

local netObj = dofile('models/milvc.lua')
local model = netObj.createModel(1)
if opt.backend == 'cudnn' then model:cuda() end
collectgarbage()

torch.save(opt.save, model)
print('==> saving milvc vgg model to ' .. opt.save)
