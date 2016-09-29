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
local normal_param = {name='normal', mw=0, stdw=0.001, mb=-6.58, stdb=0}
local gauss_param = {name='normal', mw=0, stdw=0.1, mb=0, stdb=0.1}
local gauss_param_2 = {name='normal', mw=0, stdw=0.01, mb=0, stdb=0.01}

local function makeMeanVar(inC, outC)
    local mean_logvar = nn.ConcatTable()
    local mod_mean = nn.Linear(inC, outC)
    initializer(mod_mean, gauss_param)
    local mod_logvar = nn.Linear(inC, outC)
    initializer(mod_logvar, gauss_param_2)
    mean_logvar:add(mod_mean)
    mean_logvar:add(mod_logvar)

    return mean_logvar
end

local function makeEncoder(inC, midC, outC)
    local s = nn.Sequential()
    s:add(nn.Linear(inC,  midC))
    s:add(nn.ReLU(true))

    local mean_logvar = makeMeanVar(midC, outC)
    s:add(mean_logvar)

    return s
end

local function makeEncoderWithFC7(probC, fc7C, outC)
    local s = nn.Sequential()

    local encoder_1 = nn.ParallelTable()
    local encoder_1_cap = nn.Linear(1000, probC)
    initializer(encoder_1_cap, gauss_param_2)
    local encoder_1_vf = nn.Linear(4096, fc7C)
    initializer(encoder_1_vf, gauss_param)
    encoder_1:add(encoder_1_cap):add(encoder_1_vf)

    s:add(encoder_1):add(nn.JoinTable(2)):add(nn.ReLU(true))

    local mean_logvar = makeMeanVar(probC+fc7C, outC)
    s:add(mean_logvar)

    return s
end

local function makeDecoder(inC, midC, outC)
    local s = nn.Sequential()
    local decoder_1 = nn.Linear(inC, midC)
    initializer(decoder_1, gauss_param)
    s:add(decoder_1):add(nn.ReLU(true))
    local decoder_2 = nn.Linear(midC, outC)
    initializer(decoder_2, normal_param)
    decoder_2.bias[{{1,nClasses/10}}]:fill(0)
    s:add(decoder_2)
    return s
end

local function weighted(inC, initW)
    local l
    if inC > 1 then
        l = nn.CMul(inC)
    else
        l = nn.Mul()
    end
    l.weight:fill(initW)
    return l
end

local function createVAE()
    local input = nn.Identity()()

    local encoders = {}
    local samplers = {}
    local decoder_scores = {} -- prediction score
    local decoder_weighted_scores = {}
    local decoders = {} -- prediction probability

    -- construct encoders
    encoders[1] = makeEncoderWithFC7(500, 2000, 2500)(input)
    samplers[1] = nn.GaussianSampler()(encoders[1])
    encoders[2] = makeEncoder(2500,2500,2500)(samplers[1])
    samplers[2] = nn.GaussianSampler()(encoders[2])
    -- construct decoders
    decoder_scores[1] = makeDecoder(2500, 2000, 1000)(samplers[1])
    decoder_scores[2] = makeDecoder(2500, 2000, 1000)(samplers[2])
    decoders[1] = nn.Sigmoid()(decoder_scores[1])
    decoders[2] = nn.Sigmoid()(decoder_scores[2])
    -- combined prediction scores
    decoder_weighted_scores[1] = weighted(1000, 0.5)(decoder_scores[1])
    decoder_weighted_scores[2] = weighted(1000, 0.5)(decoder_scores[2])

    local combine_scores = nn.CAddTable()(decoder_weighted_scores)
    local combine_pred = nn.Sigmoid()(combine_scores)

    -------------------------------------------------------

    collectgarbage()

    local vae = nn.Sequential():add(nn.gModule({input}, {combine_pred, decoders[1], decoders[2]}))

    return vae
end

local function loadVAE()
    assert(paths.filep(opt.vae))
    local input = nn.Identity()()

    local encoders = {}
    local samplers = {}
    local decoder_scores = {} -- prediction score
    local decoder_weighted_scores = {}
    local decoders = {} -- prediction probability

    -- construct encoders
    local prev_vae = removeParallelTb(opt.vae)
    prev_vae = prev_vae:get(1) -- nngraph
    local prev_enc1 = prev_vae:get(2)
    local prev_dec1 = prev_vae:get(4)
    local prev_enc2 = prev_vae:get(6)
    local prev_dec2 = prev_vae:get(8)
    encoders[1] = prev_enc1(input)
    samplers[1] = nn.GaussianSampler()(encoders[1])
    encoders[2] = prev_enc2(samplers[1])
    samplers[2] = nn.GaussianSampler()(encoders[2])
    -- construct decoders
    decoder_scores[1] = prev_dec1(samplers[1])
    decoder_scores[2] = prev_dec2(samplers[2])
    decoders[1] = nn.Sigmoid()(decoder_scores[1])
    decoders[2] = nn.Sigmoid()(decoder_scores[2])
    -- combined prediction scores
    decoder_weighted_scores[1] = weighted(1000, 0.5)(decoder_scores[1])
    decoder_weighted_scores[2] = weighted(1000, 0.5)(decoder_scores[2])

    local combine_scores = nn.CAddTable()(decoder_weighted_scores)
    local combine_pred = nn.Sigmoid()(combine_scores)

    -------------------------------------------------------

    collectgarbage()

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
end

function NET.createModel(nGPU)
    NET.packages()

    -- 1.1 load pre-trained milvc-vgg
    local cnn = removeParallelTb(opt.milvc_vgg):get(1)

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
    local vae
    if opt.vae and paths.filep(opt.vae) then
        vae = loadVAE()
    else
        vae = createVAE()
    end

    local format_split = nn.ParallelTable()
    format_split:add(nn.View(-1, nClasses)):add(nn.View(-1, 4096))

    -------------------------------------------------------

    collectgarbage()

    -- 1.3 Combine 1.1 and 1.2 to produce final model
    local milvc_vae = nn.Sequential():add(milvc):add(format_split):add(vae)
    local model = makeDataParallel(milvc_vae, nGPU, NET)

    model:cuda()

    return model

end

local function criterionWeight()
    local weights = torch.Tensor(nClasses):fill(1e-3)
    weights[{{nClasses/10+1,nClasses}}]:fill(1)
    return weights
end

function NET.createCriterion()
    local criterion = nn.ParallelCriterion()
    criterion:add(nn.BCECriterion(), 20)
    criterion:add(nn.BCECriterion(), 20)
    criterion:add(nn.BCECriterion(), 20)
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
   local fc_out = outputs
   local err = criterion:forward(outputs, {labels, labels, labels})
   return outputs, err
end

function NET.gradProcessing(createdMode, modelPa, modelGradPa, currentEpoch)
    for ind=1,32 do modelGradPa[ind]:zero() end
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
    -- local minVal = 5 -- 1.5e-4
    -- local maxVal = 4 -- 1.5e-4
    -- local ExpectedTotalEpoch = 5
    -- return {LR= 1.5 * 10^-(math.min((currentEpoch-1), 4)*(minVal-maxVal)/(ExpectedTotalEpoch-1)+4),
    --         WD= 5e-4}
    return {LR=1.5e-4, WD=5e-4}
end

return NET
