require 'cudnn'
require 'loadcaffe'
require 'image'

local deployTxtPath = 'checkpoint/imagenet/vgg/deploy.prototxt'
local caffeModelPath = 'checkpoint/imagenet/vgg/vgg16.caffemodel'

-- Load by cudnn
modelCaffe = loadcaffe.load(deployTxtPath, caffeModelPath, 'cudnn')
local modules = modelCaffe:listModules()
-- list all fully connected modules
local fcs = modelCaffe:findModules('nn.Linear')
local fc_names = {}
for ind=1,#fcs do fc_names[fcs[ind].name] = true end
-- obtain nn.View layers
conv_alexnet = nn.Sequential()
local isStop = false
local viewInd = -1
for ind=2,#modules do
    if modules[ind].name == 'torch_view' then
        viewInd = ind
        break
    else
        conv_alexnet:add(modules[ind]:clone())
    end
end

-- Create a clone model
local nInputPlane = 3
fc_model = nn.Sequential()
fcn_model = nn.Sequential()
for ind=2,#modules do -- Skip first layer which denotes torch container
    local lyCaffe = modules[ind]
    local lyWeight = lyCaffe.weight
    local lyBias = lyCaffe.bias
    local nOutputPlane = lyWeight and lyWeight:size(1) or nil
    if string.sub(lyCaffe.name, 1, 2) ~= 'dr' then
        if fc_names[lyCaffe.name] then
            local iH = math.sqrt(lyWeight:nElement() / nOutputPlane / nInputPlane)
            local iW = iH
            fc2conv = cudnn.SpatialConvolution(
                nInputPlane,
                nOutputPlane,
                iH, iW, 1, 1, 0, 0
                )
            print(nOutputPlane, nInputPlane, iH, iW)
            fc2conv.weight = lyWeight:reshape(nOutputPlane, nInputPlane, iH, iW)
            fc2conv.bias = lyBias:clone()
            fcn_model:add(fc2conv)
        elseif ind > viewInd then
            fcn_model:add(lyCaffe:clone())
        end
        if ind >= viewInd then
            fc_model:add(lyCaffe:clone())
        end
        if nOutputPlane then nInputPlane = nOutputPlane end
    end
end
fcn_model:add(nn.View(nInputPlane))

conv_alexnet:cuda()
fc_model:cuda()
fcn_model:cuda()

-- Test if two model output same value
local input = image.lena()
input = image.scale(input, 224, 224)
input = input:cuda()
output_fc = fc_model:forward(conv_alexnet:forward(input))
output_fcn = fcn_model:forward(conv_alexnet:forward(input))