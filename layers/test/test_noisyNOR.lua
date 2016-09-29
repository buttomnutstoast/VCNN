require 'nn'
require 'cutorch'
if not nn.noisyNOR then dofile('layers/noisyNOR.lua') end

local function checkEq(t1, t2)
    -- t1, t2 are tensors
    assert(torch.type(t1)==torch.type(t2), 'checkEq: input tensors type inconsistent!!')
    local diff = torch.max(torch.abs(torch.csub(t1:float(),t2:float())))
    assert(diff<1e-5, ('err:%f, test failed!!!').format(diff))
end

test = {}
test.timer = torch.Timer()
function test.forward()
    test.timer:reset()
    test.model = test.model or nn.noisyNOR()
    test.model:cuda()

    test.input = test.input or torch.Tensor(10,1000,12,12):uniform():cuda()

    test.gt_output = test.input:clone():mul(-1):add(1):view(10,1000,-1):prod(3):squeeze(3):mul(-1):add(1)
    local max_pool = test.input:clone():max(4):max(3)
    test.gt_output:cmax(max_pool)
    test.output = test.model:forward(test.input)
    print(('forward takes %s s').format(test.timer:time().real))
    checkEq(test.output, test.gt_output)
end


function test.backward()
    test.timer:reset()
    test.gradOutput = test.gradOutput or torch.Tensor(10,1000):uniform():cuda()
    test.gradInput = test.model:updateGradInput(test.input, test.gradOutput)
    assert(torch.all(torch.eq(test.gradInput, test.gradInput)), 'nan in test.gradInput!!!!')
    test.gt_gradInput = test.output:view(10,1000,1,1):expand(10,1000,12,12):clone()
    test.gt_gradInput:mul(-1):add(1)
    local tmp = test.input:clone():mul(-1):add(1)
    test.gt_gradInput:cdiv(tmp)
    test.gt_gradInput:maskedFill(torch.gt(test.gt_gradInput, 1),1)
    test.gt_gradInput:cmul(test.gradOutput:view(10,1000,1,1):expand(10,1000,12,12))

    print(('backward takes %s s').format(test.timer:time().real))
    checkEq(test.gradInput, test.gt_gradInput)
end

for i=1,100 do
    test.forward()
    test.backward()
    print('test passed!')
end