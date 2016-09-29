local M = {}

function M.caffe2torch(deployTxtPath, caffeModelPath, backend, convertLinear2Conv)
    bk = require(backend)
    require 'loadcaffe'
    require 'image'

    -- Load by cudnn
    modelCaffe = loadcaffe.load(deployTxtPath, caffeModelPath, backend)
    if not convertLinear2Conv then
        return modelCaffe
    else
        local modules = modelCaffe:listModules()
        -- list all fully connected modules
        local fcs = modelCaffe:findModules('nn.Linear')
        local fc_names = {}
        for ind=1,#fcs do fc_names[fcs[ind].name] = true end
        -- list layers which determine the output dimension
        local convs = modelCaffe:findModules(backend..'.SpatialConvolution')
        local fc_conv_names = {}
        for ind=1,#convs do fc_conv_names[convs[ind].name] = true end
        for ind=1,#fcs do fc_conv_names[fcs[ind].name] = true end
        -- obtain nn.View layers
        local views = modelCaffe:findModules('nn.View')
        local view_names = {}
        for ind=1,#views do view_names[views[ind].name] = true end

        -- Create a clone model
        alexnet = nn.Sequential()
        local nInputPlane = 3
        for ind=2,#modules do -- Skip first layer which denotes torch container
            local lyCaffe = modules[ind]
            local lyWeight = lyCaffe.weight
            local lyBias = lyCaffe.bias
            local nOutputPlane = (lyWeight ~= nil) and lyWeight:size(1) or -1
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
                alexnet:add(fc2conv)
            elseif not view_names[lyCaffe.name] then
                alexnet:add(lyCaffe:clone())
            end
            if fc_conv_names[lyCaffe.name] then
                nInputPlane = nOutputPlane
            end
        end
        -- alexnet:add(nn.View(nInputPlane))
        return alexnet
    end
end

return M.caffe2torch