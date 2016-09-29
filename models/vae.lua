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

local function makeShallowEncoderWithFC7(outC)
    local s = nn.Sequential()

    local mean_logvar = makeMeanVar(1000+4096, outC)
    s:add(nn.JoinTable(2)):add(mean_logvar)

    return s
end

local function makeShallowDecoder(inC, outC)
    local s = nn.Sequential()
    local decoder_2 = nn.Linear(inC, outC)
    initializer(decoder_2, normal_param)
    decoder_2.bias[{{1,nClasses/10}}]:fill(0)
    s:add(decoder_2)
    return s
end

local NET = {}
function NET.packages()
    if not nn.GaussianSampler then dofile('layers/GaussianSampler.lua') end
    if not nn.VAE_KLDCriterion then dofile('layers/VAE_KLDCriterion.lua') end
end

function NET.createModel(nGPU)
    NET.packages()

    -- local encoder = makeEncoderWithFC7(500, 2000, 2500)
    local encoder = makeShallowEncoderWithFC7(2500)
    local latent_sample = nn.GaussianSampler()
    local decoder_1 = makeDecoder(2500, 2000, 1000)
    local decoder_2 = makeDecoder(2500, 2000, 4096)

    -- Split two outputs {encoder_output, decoder_output}
    local split_output = nn.ConcatTable()
    local encoder_output = nn.Identity()
    local decoder1_output = nn.Sequential():add(latent_sample):add(decoder_1):add(nn.Sigmoid())
    local decoder2_output = nn.Sequential():add(latent_sample):add(decoder_2):add(nn.ReLU())
    split_output:add(decoder1_output):add(decoder2_output):add(encoder_output)

    -- Construct complete model
    local vae = nn.Sequential():add(encoder):add(split_output)
    local model = makeDataParallel(vae, nGPU, NET)
    model:cuda()

    collectgarbage()

    return model
end

function NET.createCriterion()
    local criterion = nn.ParallelCriterion()
    criterion:add(nn.BCECriterion(), 10)
    criterion:add(nn.SmoothL1Criterion(), 10)
    criterion:add(nn.VAE_KLDCriterion(), 1e-5)
    return criterion
end

function NET.trainOutputInit()
    local info = {}
    info[#info+1] = newInfoEntry('loss',0,0)
    info[#info+1] = newInfoEntry('map',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err)
    batch_size = outputs[1]:size(1)
    predictions = outputs[1]:float()
    labels = labelsCPU[1]

    info[1].value   = err
    info[1].N       = batch_size

    info[2].value   = meanAvgPrec(predictions, labels, 2, opt.threshold)
    info[2].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = newInfoEntry('mil_prob',0,0, true)
    info[#info+1] = newInfoEntry('map',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    batch_size = outputs[1]:size(1)
    predictions = outputs[1]:float()
    labels = labelsCPU[1]

    info[1].value = predictions
    info[2].value = meanAvgPrec(predictions, labels, 2, opt.threshold)
    info[2].N     = batch_size
end

function NET.gradProcessing(createdMode, modelPa, modelGradPa, currentEpoch)
    -- processing!!!
end

function NET.feval(inputs, labels)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, {labels[1], labels[2], {}})
   local gradOutputs = criterion:backward(outputs, {labels[1], labels[2], {}})
   model:backward(inputs, gradOutputs)
   return outputs, err
end

function NET.ftest(inputs, labels)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, {labels[1], labels[2], {}})
   return outputs, err
end

function NET.arguments(cmd)
    cmd:option('-vocabs', 'vocabs/vocab_words.txt','Path to the file of 1000 common vocabs in MSCOCO')
    cmd:option('-threshold', 0.1, 'threshold for predictions')
end

function NET.trainRule(currentEpoch)
    return {LR=1e-3, WD=5e-4}
end

return NET
