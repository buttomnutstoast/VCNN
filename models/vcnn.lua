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

local function createVAE()
    local encoder = makeEncoderWithFC7(500, 2000, 2500)
    local latent_sample = nn.GaussianSampler()
    local decoder_1 = makeDecoder(2500, 2000, 1000)

    -- Construct complete model
    local vae = nn.Sequential():add(encoder):add(latent_sample):add(decoder_1):add(nn.Sigmoid())

    return vae
end

local function loadVae()
    assert(paths.filep(opt.vae))
    local prev_vae = removeParallelTb(opt.vae)
    local prev_enc = prev_vae:get(1)
    local prev_dec = prev_vae:get(2)
    local encoder = prev_enc:clone()
    local prev_dec_cap = prev_dec:get(1):get(2)
    local decoder_cap = prev_dec_cap:clone()
    local vae = nn.Sequential():add(encoder):add(nn.GaussianSampler()):add(decoder_cap):add(nn.Sigmoid())
    return vae
end

local NET = {}
function NET.packages()
    if not nn.ConstantAdd then dofile('layers/ConstantAdd.lua') end
    if not nn.AddSingletonDim then dofile('layers/AddSingletonDim.lua') end
    if not nn.RemoveLastSingleton then dofile('layers/RemoveLastSingleton.lua') end
    if not nn.GaussianSampler then dofile('layers/GaussianSampler.lua') end
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
        vae = loadVae()
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
    local weights = torch.Tensor(nClasses):fill(1)
    weights[{{1, nClasses/10}}]:fill(1e-3)
    return weights
end

function NET.createCriterion()
    local criterion = nn.MultiCriterion()
    criterion:add(nn.BCECriterion(criterionWeight()), 20)
    return criterion
end

function NET.trainOutputInit()
    local info = {}
    info[#info+1] = newInfoEntry('loss',0,0)
    info[#info+1] = newInfoEntry('map',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err)
    batch_size = outputs:size(1)
    predictions = outputs:float()
    labels = labelsCPU

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
    batch_size = outputs:size(1)
    predictions = outputs:float()
    labels = labelsCPU

    info[1].value = predictions
    info[2].value = meanAvgPrec(predictions, labels, 2, opt.threshold)
    info[2].N     = batch_size
end

function NET.gradProcessing(createdMode, modelPa, modelGradPa, currentEpoch)
    -- for ind=1,32 do modelGradPa[ind]:zero() end
end

function NET.arguments(cmd)
    cmd:option('-vocabs', 'vocabs/vocab_words.txt','Path to the file of 1000 common vocabs in MSCOCO')
    cmd:option('-rMean', '123.68', 'mean pixel value of channel R')
    cmd:option('-gMean', '116.779', 'mean pixel value of channel G')
    cmd:option('-bMean', '103.939', 'mean pixel value of channel B')
    cmd:option('-milvc_vgg', 'checkpoint/milvc/milvc_vgg.t7', 'Path to the trained milvc vgg model')
    cmd:option('-vae', '/path/to/trained/vae/model', 'path to the pre-trained vae model')
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
