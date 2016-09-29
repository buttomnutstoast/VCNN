local function linearToConv(mod)
    -- Check outermost module type of model
    local recur = {}
    recur['nn.Sequential'] = true
    recur['nn.ConcatTable'] = true
    recur['nn.ParallelTable'] = true

    local mod_type = torch.type(mod)
    assert(recur[mod_type],
        'input module should be rather nn.Sequential, nn.ConcatTable or nn.ParallelTable')
    mod_type = mod_type:sub(4, #mod_type)
    local fc2Conv = nn[mod_type]()

    for i, sub_mod in ipairs(mod.modules) do
        local submod_type = torch.type(sub_mod)
        local newmod
        if recur[submod_type] then
            newmod = linearToConv(sub_mod)
        elseif submod_type == 'nn.Linear' then
            local nIn, nOut = sub_mod.weight:size(2), sub_mod.weight:size(1)
            newmod = nn.SpatialConvolution(nIn, nOut, 1, 1, 1, 1, 0, 0):cuda()
            newmod.weight = sub_mod.weight:clone():viewAs(newmod.weight)
            newmod.bias = sub_mod.bias:clone()
        else
            newmod = sub_mod:clone()
        end
        fc2Conv:add(newmod:clone())
    end
    return fc2Conv
end

local function cudnnToNN(mod)
    local recur = {}
    recur['nn.Sequential'] = true
    recur['nn.ConcatTable'] = true
    recur['nn.ParallelTable'] = true
    recur['nn.gModule'] = true

    assert(recur[torch.type(mod)],
        'input module should be rather nn.Sequential, nn.ConcatTable, '
        .. ' nn.ParallelTable or nn.gModule')

    for ind, sub_mod in ipairs(mod.modules) do
        if recur[torch.type(sub_mod)] then
            cudnnToNN(sub_mod)
        elseif torch.type(sub_mod) == 'cudnn.SpatialConvolution' then
            newmod = nn.SpatialConvolution(
                sub_mod.nInputPlane,
                sub_mod.nOutputPlane,
                sub_mod.kW,
                sub_mod.kH,
                sub_mod.dW,
                sub_mod.dH,
                sub_mod.padW,
                sub_mod.padH
                ):cuda()
            newmod.weight:copy(sub_mod.weight)
            newmod.bias:copy(sub_mod.bias)
            mod.modules[ind] = newmod
        elseif torch.type(sub_mod) == 'cudnn.ReLU' then
            mod.modules[ind] = nn.ReLU()
        elseif torch.type(sub_mod) == 'cudnn.Sigmoid' then
            mod.modules[ind] = nn.Sigmoid()
        elseif torch.type(sub_mod) == 'cudnn.SpatialMaxPooling' then
            newmod = nn.SpatialMaxPooling(
                sub_mod.kW,
                sub_mod.kH,
                sub_mod.dW,
                sub_mod.dH,
                sub_mod.padW,
                sub_mod.padH
                )
            mod.modules[ind] = newmod
        end
    end
end

local function initializer(module, method)
    --- Helper to initialize module weights with specified methods
    assert(method.name, 'Please specify initializer in method.name')
    if method.name == 'xavier' then
        method.constant = method.constant or 1
        method.sqr_const = method.sqr_const or 1
        local function xavier(weight)
            local wIn, wOut = weight:size(2), weight:size(1) -- row-centric matrix manipulation
            local val = method.constant * torch.sqrt(method.sqr_const / (wIn + wOut))
            weight:uniform(-val, val)
        end
        xavier(module.weight)
        module.bias:zero()
    elseif method.name == 'normal' then
        method.mw = method.mw or 0
        method.stdw = method.stdw or 0.01
        method.mb = method.mb or 0
        method.stdb = method.stdb or 0.01
        if method.stdw == 0 then
            module.weight:fill(method.mw)
        else
            module.weight:normal(method.mw, method.stdw)
        end
        if method.stdb == 0 then
            module.bias:fill(method.mb)
        else
            module.bias:normal(method.mb, method.stdb)
        end
    else
        error('Unknown weight initialization method ' .. method.name)
    end
end

-- initilization template
local xavier_param = {name='xavier', constant=1, sqr_const=2}
local normal_param = {name='normal', mw=0, stdw=0.001, mb=-6.58, stdb=0}
local gauss_param = {name='normal', mw=0, stdw=0.1, mb=0, stdb=0.1}
local gauss_param_2 = {name='normal', mw=0, stdw=0.01, mb=0, stdb=0.01}

local function makeMeanVar(inC, outC)
    local mean_logvar = nn.ConcatTable()
    local mod_mean = cudnn.SpatialConvolution(inC, outC,1,1,1,1,0,0)
    initializer(mod_mean, gauss_param)
    local mod_logvar = cudnn.SpatialConvolution(inC, outC,1,1,1,1,0,0)
    initializer(mod_logvar, gauss_param_2)
    mean_logvar:add(mod_mean)
    mean_logvar:add(mod_logvar)

    return mean_logvar
end

local function makeEncoder(inC, midC, outC)
    local s = nn.Sequential()
    s:add(cudnn.SpatialConvolution(inC,  midC,1,1,1,1,0,0))
    s:add(nn.ReLU(true))

    local mean_logvar = makeMeanVar(midC, outC)
    s:add(mean_logvar)

    return s
end

local function makeEncoderWithFC7(probC, fc7C, outC)
    local s = nn.Sequential()

    local encoder_1 = nn.ParallelTable()
    local encoder_1_cap = cudnn.SpatialConvolution(1000, probC,1,1,1,1,0,0)
    initializer(encoder_1_cap, gauss_param_2)
    local encoder_1_vf = cudnn.SpatialConvolution(4096, fc7C,1,1,1,1,0,0)
    initializer(encoder_1_vf, gauss_param)
    encoder_1:add(encoder_1_cap):add(encoder_1_vf)

    s:add(encoder_1):add(nn.JoinTable(2)):add(nn.ReLU(true))

    local mean_logvar = makeMeanVar(probC+fc7C, outC)
    s:add(mean_logvar)

    return s
end

local function makeDecoder(inC, midC, outC)
    local s = nn.Sequential()
    local decoder_1 = cudnn.SpatialConvolution(inC, midC,1,1,1,1,0,0)
    initializer(decoder_1, gauss_param)
    s:add(decoder_1):add(nn.ReLU(true))
    local decoder_2 = cudnn.SpatialConvolution(midC, outC,1,1,1,1,0,0)
    initializer(decoder_2, normal_param)
    decoder_2.bias[{{1,nClasses/10}}]:fill(0)
    s:add(decoder_2)
    return s
end

local function weighted(inC, initW)
    local l
    if inC > 1 then
        l = nn.CMulExpand(1, inC, 1, 1)
    else
        l = nn.Mul()
    end
    l.weight:fill(initW)
    return l
end

local function createVAE(encoder_mods, decoder_mods, decoder_weights)
    local input = nn.Identity()()

    local encoders = {}
    local samplers = {}
    local decoder_scores = {} -- prediction score
    local decoder_weighted_scores = {}
    local decoders = {} -- prediction probability

    -- initialize encoder, decoder mods
    encoder_mods = encoder_mods or {makeEncoderWithFC7(500,2000,2500), makeEncoder(2500,2500,2500)}
    decoder_mods = decoder_mods or {makeDecoder(2500,2000,1000), makeDecoder(2500,2000,1000)}
    decoder_weight_mods = {weighted(1000, 0.5), weighted(1000, 0.5)}

    -- get pre-trained weight from decoder_weights
    decoder_weight_mods[1].weight = decoder_weights[1].weight:clone():resizeAs(decoder_weight_mods[1].weight)
    decoder_weight_mods[2].weight = decoder_weights[2].weight:clone():resizeAs(decoder_weight_mods[2].weight)

    -- construct encoders
    encoders[1] = encoder_mods[1](input)
    samplers[1] = nn.GaussianSampler()(encoders[1])
    encoders[2] = encoder_mods[2](samplers[1])
    samplers[2] = nn.GaussianSampler()(encoders[2])
    -- construct decoders
    decoder_scores[1] = decoder_mods[1](samplers[1])
    decoder_scores[2] = decoder_mods[2](samplers[2])
    decoders[1] = nn.Sigmoid()(decoder_scores[1])
    decoders[2] = nn.Sigmoid()(decoder_scores[2])
    -- combined prediction scores
    decoder_weighted_scores[1] = decoder_weight_mods[1](decoder_scores[1])
    decoder_weighted_scores[2] = decoder_weight_mods[2](decoder_scores[2])

    local combine_scores = nn.CAddTable()(decoder_weighted_scores)
    local combine_pred = nn.Sigmoid()(combine_scores)

    -------------------------------------------------------

    local vae = nn.Sequential():add(nn.gModule({input}, {combine_pred, decoders[1], decoders[2]}))

    return vae
end

local NET = {}
function NET.packages()
    require 'nngraph'
    if not nn.ConstantAdd then dofile('layers/ConstantAdd.lua') end
    if not nn.AddSingletonDim then dofile('layers/AddSingletonDim.lua') end
    if not nn.RemoveLastSingleton then dofile('layers/RemoveLastSingleton.lua') end
    if not nn.GaussianSampler then dofile('layers/GaussianSampler.lua') end
    if not nn.TableCat then dofile('layers/TableCat.lua') end
    if not nn.CMulExpand then dofile('layers/CMulExpand.lua') end
    if not nn.noisyNOR then dofile('layers/noisyNOR.lua') end
end

function NET.createModel(nGPU)
    NET.packages()

    -- 1.1 load pre-trained milvc-vgg
    local cnn = removeParallelTb(opt.milvc_vgg):get(1)
    cudnnToNN(cnn)

    -- decouple fc8 from cnn
    local fc8 = nn.Sequential():add(cnn:get(38)):add(cnn:get(39))
    cnn:remove(39)
    cnn:remove(38)
    -- remove dropout layer
    cnn:remove(37)
    cnn:remove(34)

    -- split two outpus {cnn_output, nor_output}
    local split_features = nn.ConcatTable()
    split_features:add(fc8):add(nn.Identity())

    local milvc = nn.Sequential():add(cnn):add(split_features)

    -- 1.2 load pre-trained VAE
    local encoder_mods, decoder_mods, decoder_weights
    if opt.vae and paths.filep(opt.vae) then
        --- list modules in the vae graph
        --  [1]: nn.Identity, [2]: encoder1, [3]: sampler1, [4]: decoder1, [5]: score_weight1
        --  [6]: encoder2,    [7]: sampler2, [8]: decoder2, [9]: score_weight2
        encoder_mods, decoder_mods, decoder_weights = {}, {}, {}
        local prev_vae = removeParallelTb(opt.vae)
        prev_vae = prev_vae:get(1)
        encoder_mods[1] = linearToConv(prev_vae:get(2))
        decoder_mods[1] = linearToConv(prev_vae:get(4))
        decoder_weights[1] = prev_vae:get(5):clone()
        encoder_mods[2] = linearToConv(prev_vae:get(6))
        decoder_mods[2] = linearToConv(prev_vae:get(8))
        decoder_weights[2] = prev_vae:get(9):clone()
    end
    local vae = createVAE(encoder_mods, decoder_mods, decoder_weights)

    -- 1.3 post-process probability distributions
    local prob_path = nn.ParallelTable()
    prob_path:add(nn.noisyNOR()):add(nn.noisyNOR()):add(nn.noisyNOR())

    -------------------------------------------------------

    collectgarbage()

    -- 1.4 Combine 1.1 and 1.2 to produce final model
    local milvc_vae = nn.Sequential():add(milvc):add(vae):add(prob_path)
    local model = makeDataParallel(milvc_vae, nGPU, NET)

    model:cuda()

    return model
end

local function criterionWeight()
    local weights = torch.Tensor(nClasses):fill(1)
    weights[{{1,nClasses/10}}]:fill(1e-3)
    return weights
end

function NET.createCriterion()
    local criterion = nn.ParallelCriterion()
    criterion:add(nn.BCECriterion(criterionWeight()), 20)
    criterion:add(nn.BCECriterion(criterionWeight()), 20)
    criterion:add(nn.BCECriterion(criterionWeight()), 20)
    return criterion
end

local function getPred(outputs)
    local proc_outputs = torch.Tensor(outputs[1]:size()):zero():float()
    proc_outputs:add(outputs[1]:float())
    proc_outputs:add(outputs[2]:float())
    proc_outputs:add(outputs[3]:float())
    proc_outputs:div(3)
    return proc_outputs
end

function NET.trainOutputInit()
    local info = {}
    info[#info+1] = newInfoEntry('loss',0,0) -- last number  = batch_size. must >0
    info[#info+1] = newInfoEntry('map-combine',0,0)
    info[#info+1] = newInfoEntry('map-1',0,0)
    info[#info+1] = newInfoEntry('map-2',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err)
    batch_size = outputs[1]:size(1)
    prediction_combine = outputs[1]:float()
    prediction_1 = outputs[2]:float()
    prediction_2 = outputs[3]:float()
    labels = labelsCPU

    info[1].value   = err*opt.iterSize
    info[1].N       = batch_size

    info[2].value   = meanAvgPrec(prediction_combine, labels, 2, opt.threshold)
    info[2].N       = batch_size

    info[3].value   = meanAvgPrec(prediction_1, labels, 2, opt.threshold)
    info[3].N       = batch_size

    info[4].value   = meanAvgPrec(prediction_2, labels, 2, opt.threshold)
    info[4].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = newInfoEntry('mil_prob',0,0, true)
    info[#info+1] = newInfoEntry('map',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    batch_size = outputs[1]:size(1)
    predictions = getPred(outputs)
    labels = labelsCPU

    info[1].value = predictions
    info[2].value = meanAvgPrec(predictions, labels, 2, opt.threshold)
    info[2].N     = batch_size
end

function NET.feval(inputs, labels)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, {labels, labels, labels})
   local gradOutputs = criterion:backward(outputs, {labels, labels, labels})
   model:backward(inputs, gradOutputs)
   return outputs, err
end

function NET.ftest(inputs, labels)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, {labels, labels, labels})
   return outputs, err
end

function NET.gradProcessing(createdMode, modelPa, modelGradPa, currentEpoch)
    --
end

function NET.arguments(cmd)
    cmd:option('-vocabs', 'vocabs/vocab_words.txt','Path to the file of 1000 common vocabs in MSCOCO')
    cmd:option('-rMean', '123.68', 'mean pixel value of channel R')
    cmd:option('-gMean', '116.779', 'mean pixel value of channel G')
    cmd:option('-bMean', '103.939', 'mean pixel value of channel B')
    cmd:option('-milvc_vgg', 'checkpoint/milvc/milvc_vgg.t7', 'Path to the trained milvc vgg model')
    cmd:option('-vae', '/path/to/trained/stackvae/model', 'path to the pre-trained stacked vae model')
    cmd:option('-threshold', 0.1, 'threshold for prediction')
end

function NET.trainRule(currentEpoch)
    local minVal = 6 -- 10^-6
    local maxVal = 5 -- 10^-5
    local ExpectedTotalEpoch = 10
    return {LR= 1.5 * 10^-((currentEpoch-1)*(minVal-maxVal)/(ExpectedTotalEpoch-1)+5),
            WD= 5e-4}
    -- return {LR=1.5e-5, WD=5e-4}
end

return NET
