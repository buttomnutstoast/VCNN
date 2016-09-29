local function feval(model, criterion, inputs, labels)
    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, labels)
    local gradOutputs = criterion:backward(outputs, labels)
    model:backward(inputs, gradOutputs)
    cutorch:synchronize()
    return outputs, err
end

dofile('util.lua')
require 'optim'
torch.manualSeed(1)

-- global variables
opt = {}
opt.GPU = 1
opt.iterSize = 1
opt.backend = 'cudnn'
opt.nGPU = 3
opt.LR = 1e-1
opt.momentum = 0
opt.weightDecay = 5e-4
nClasses = 1000

optimState = {
  learningRate = opt.LR,
  learningRateDecay = 0.0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}

local netObj = dofile('models/capae_cudnn.lua')
local model = netObj.createModel(1)
local model_nGPU = nn.Sequential():add(makeDataParallel(model, opt.nGPU, netObj))
local criterion = netObj.createCriterion():cuda()

-- check if weights and biases are copied!
local parameters, gradParameters = model:getParameters()
local parameters_nGPU, gradParameters_nGPU = model_nGPU:getParameters()
model:zeroGradParameters()
model_nGPU:zeroGradParameters()
assert(torch.max(torch.abs(torch.csub(parameters:float(), parameters_nGPU:float()))) < 1e-5, 'inconsisten paramters')
assert(torch.max(torch.abs(torch.csub(gradParameters:float(), gradParameters_nGPU:float()))) < 1e-5, 'inconsisten gradparamters')

for epoch=1, 3 do
    print(('epoch:%d'):format(epoch))
    -- construct input tensor
    local inputs = torch.Tensor(4, nClasses):uniform():cuda()

    local output, err = feval(model, criterion, inputs, inputs)
    local output_nGPU, err_nGPU = feval(model_nGPU, criterion, inputs, inputs)
    -- -- check if two modules output same value
    assert(torch.max(torch.abs(torch.csub(output:float(), output_nGPU:float()))) < 1e-5, 'inconsistent output!')
    assert(math.max(math.abs(err - err_nGPU)) < 1e-5, 'inconsistent err!')

    -- -- check if gradParameters are the same
    assert(torch.max(torch.abs(torch.csub(gradParameters:float(), gradParameters_nGPU:float()))) < 1e-5, 'inconsisten paramters')

    optim.sgd(function() return err, gradParameters end, parameters, optimState)
    optim.sgd(function() return err_nGPU, gradParameters_nGPU end, parameters_nGPU, optimState)
    if model_nGPU.needsSync then
        model_nGPU:syncParameters()
    end
    assert(torch.max(torch.abs(torch.csub(parameters:float(), parameters_nGPU:float()))) < 1e-5, 'inconsisten paramters')
end