require 'nn'
require 'cutorch'
if not nn.CMulExpand then dofile('layers/CMulExpand.lua') end

local function checkEq(t1, t2)
    -- t1, t2 are tensors
    assert(torch.type(t1)==torch.type(t2), 'checkEq: input tensors type inconsistent!!')
    assert(torch.max(torch.abs(torch.csub(t1:float(),t2:float())))<1e-5, 'test failed!!!')
end

test = {}
function test.forward()
    test.model = test.model or nn.CMulExpand(1,3,1,1)
    local model_weight = test.model.weight

    test.input = test.input or torch.Tensor(3,3,3,3):uniform()
    local weight = model_weight:expandAs(test.input)

    local gt_output = torch.cmul(test.input, weight)
    local output = test.model:forward(test.input)
    checkEq(output, gt_output)
end

function test.forward_cuda()
    test.cuda_model = test.cuda_model or nn.CMulExpand(1,3,1,1)
    test.cuda_model:cuda()
    local model_weight = test.cuda_model.weight

    test.cuda_input = test.input:cuda() or torch.Tensor(3,3,3,3):uniform():cuda()
    local weight = model_weight:expandAs(test.cuda_input)

    local gt_output = torch.cmul(test.cuda_input, weight)
    local output = test.cuda_model:forward(test.cuda_input)
    checkEq(output, gt_output)
end

function test.backward()
    test.gradOutput = test.gradOutput or torch.Tensor(3,3,3,3):uniform()
    test.model:backward(test.input, test.gradOutput)
    local weight = test.model.weight:clone()
    local gt_gradWeight = torch.Tensor(weight:size()):zero()
    for iN=1,3 do
        for iH=1,3 do
            for iW=1,3 do
                local sIn = test.input[{{iN},{},{iH},{iW}}]
                local sGradOutput = test.gradOutput[{{iN},{},{iH},{iW}}]
                gt_gradWeight:add(torch.cmul(sIn, sGradOutput))
            end
        end
    end
    checkEq(test.model.gradWeight, gt_gradWeight)
end

function test.backward_cuda()
    test.cuda_gradOutput = test.cuda_gradOutput or torch.Tensor(3,3,3,3):uniform():cuda()
    test.cuda_model:backward(test.cuda_input, test.cuda_gradOutput)
    local weight = test.cuda_model.weight:clone()
    local gt_gradWeight = torch.Tensor(weight:size()):zero()
    for iN=1,3 do
        for iH=1,3 do
            for iW=1,3 do
                local sIn = test.input[{{iN},{},{iH},{iW}}]
                local sGradOutput = test.gradOutput[{{iN},{},{iH},{iW}}]
                gt_gradWeight:add(torch.cmul(sIn, sGradOutput))
            end
        end
    end
    checkEq(test.model.gradWeight, gt_gradWeight)
end

test.forward()
test.forward_cuda()
test.backward()
test.backward_cuda()
print('test passed!')