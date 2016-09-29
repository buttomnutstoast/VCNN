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

local function makeProbGate()
    return nn.View(-1, nClasses)
end

local function loadPrevVae()
    assert(paths.filep(opt.prev_vae))
    local prev_vae = removeParallelTb(opt.prev_vae)
    local prev_enc = prev_vae:get(1)
    local prev_dec = prev_vae:get(2)
    local encoder = prev_enc:clone()
    local prev_dec_cap = prev_dec:get(1):get(2)
    local prev_dec_vf = prev_dec:get(2):get(2)
    local decoder_cap = prev_dec_cap:clone()
    local decoder_vf = prev_dec_vf:clone()
    return encoder, decoder_cap, decoder_vf
end

local NET = {}
function NET.packages()
    require 'nngraph'
    if not nn.ConstantAdd then dofile('layers/ConstantAdd.lua') end
    if not nn.AddSingletonDim then dofile('layers/AddSingletonDim.lua') end
    if not nn.RemoveLastSingleton then dofile('layers/RemoveLastSingleton.lua') end
    if not nn.GaussianSampler then dofile('layers/GaussianSampler.lua') end
    if not nn.VAE_KLDCriterion then dofile('layers/VAE_KLDCriterion.lua') end
    if not nn.TableCat then dofile('layers/TableCat.lua') end
end

function NET.createModel(nGPU)
    NET.packages()

    local input = nn.Identity()()

    local encoders = {}
    local mean_logvar_tbs = {} -- {{{mean1, std1}}, {{mean2, std2}},...}
    local samplers = {}
    local decoder_scores = {} -- prediction score
    local rec_scores = {} -- reconstruction score
    local decoder_weighted_scores = {}
    local decoders = {} -- prediction probability
    local recers = {}

    -- construct encoders
    local prev_enc, prev_dec_cap, prev_dec_vf = loadPrevVae()
    encoders[1] = prev_enc(input)
    mean_logvar_tbs[1] = nn.ConcatTable():add(nn.Identity())(encoders[1])
    samplers[1] = nn.GaussianSampler()(encoders[1])
    encoders[2] = makeEncoder(2500,2500,2500)(samplers[1])
    mean_logvar_tbs[2] = nn.ConcatTable():add(nn.Identity())(encoders[2])
    samplers[2] = nn.GaussianSampler()(encoders[2])
    -- construct decoders
    decoder_scores[1] = prev_dec_cap(samplers[1])
    decoder_scores[2] = makeDecoder(2500, 2000, 1000)(samplers[2])
    decoders[1] = nn.Sigmoid()(decoder_scores[1])
    decoders[2] = nn.Sigmoid()(decoder_scores[2])
    -- construct visual feature reconstructor
    rec_scores[1] = prev_dec_vf(samplers[1])
    rec_scores[2] = makeDecoder(2500, 2000, 4096)(samplers[2])
    recers[1] = nn.ReLU()(rec_scores[1])
    recers[2] = nn.ReLU()(rec_scores[2])
    -- combined prediction scores
    decoder_weighted_scores[1] = weighted(1000, 0.5)(decoder_scores[1])
    decoder_weighted_scores[2] = weighted(1000, 0.5)(decoder_scores[2])

    local combine_scores = nn.CAddTable()(decoder_weighted_scores)
    local combine_pred = nn.Sigmoid()(combine_scores)

    -- flatten mean_logvar_tbs to {{mean1, std1}, {mean2, std2},...}
    local flat_mean_logvars = nn.TableCat()(mean_logvar_tbs)

    -------------------------------------------------------

    collectgarbage()

    local vae = nn.Sequential():add(nn.gModule({input}, {combine_pred, decoders[1], decoders[2], recers[1], recers[2], flat_mean_logvars}))
    local model = makeDataParallel(vae, nGPU, NET)
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
    criterion:add(nn.BCECriterion(), 10)
    criterion:add(nn.BCECriterion(), 10)
    criterion:add(nn.BCECriterion(), 10)
    criterion:add(nn.SmoothL1Criterion(), 10)
    criterion:add(nn.SmoothL1Criterion(), 10)
    local ALotOfKLDs = nn.ParallelCriterion()
    ALotOfKLDs:add(nn.VAE_KLDCriterion(), 1e-5)
    ALotOfKLDs:add(nn.VAE_KLDCriterion(), 1e-5)
    criterion:add(ALotOfKLDs,1)
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
    labels = labelsCPU[1]

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
    labels = labelsCPU[1]

    info[1].value = predictions
    info[2].value = meanAvgPrec(predictions, labels, 2, opt.threshold)
    info[2].N     = batch_size
end

function NET.evalOutputInit()
    local info = {}
    info[#info+1] = newInfoEntry('map',0,0)
    return info
end

function NET.evalOutput(info, outputs, labelsCPU, err)
    batch_size = outputs[1]:size(1)
    predictions = getPred(outputs)
    labels = labelsCPU[1]

    info[1].value = meanAvgPrec(predictions, labels, 2, opt.threshold)
    info[1].N     = batch_size
end

function NET.feval(inputs, labels)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, {labels[1], labels[1], labels[1], labels[2], labels[2], {}})
   local gradOutputs = criterion:backward(outputs, {labels[1], labels[1], labels[1], labels[2], labels[2], {}})
   model:backward(inputs, gradOutputs)
   return outputs, err
end

function NET.ftest(inputs, labels)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, {labels[1], labels[1], labels[1], labels[2], labels[2], {}})
   return outputs, err
end

function NET.gradProcessing(createdMode, modelPa, modelGradPa, currentEpoch)
    --- list modules in the graph
    --  [1]: nn.Identity, [2]: encoder1, [3]: sampler1, [4]: decoder1, [5]: AddSingletonDim1
    --  [6]: encoder2,    [7]: sampler2, [8]: decoder2, [9]: AddSingletonDim12
    --  list number of parameters of each module in the graph
    --  [1-8]: encoder1, [9~12]: decoder1, [13]: decoder1_weight
    --  [14-19]: encoder2, [20-23]: decoder2, [24]: decoder2_weight
    -- for ind=1,12 do modelGradPa[ind]:mul(0.1) end
end

function NET.arguments(cmd)
    cmd:option('-vocabs', 'vocabs/vocab_words.txt','Path to the file of 1000 common vocabs in MSCOCO')
    cmd:option('-threshold', 0.1, 'threshold for predictions')
    cmd:option('-prev_vae', '/path/to/trained/vae/model')
end

function NET.trainRule(currentEpoch)
    return {LR=1e-3, WD=5e-4}
end

return NET
