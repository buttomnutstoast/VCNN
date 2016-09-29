require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

paths.dofile('processMSCOCO.lua')

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
      A dataset class for images in a flat folder structure (folder-name is
      class-name). Optimized for extremely large datasets (upwards of 14
      million images). Tested only on Linux (as it uses command-line linux
      utilities to scale up)
    ]],
    {name="path",
     type="string",
     help="Path to root directory of MSCOCO"},

    {name="vocPath",
     type="string",
     help="path to the file of 1000 common vocabs in MSCOCO, "
          .. "we also define the class index based on the file."},

     {name="imageIdPath",
      type="string",
      help="path to the file of sampled image ids",
      opt=true},

    {name="protocol",
     type="string",
     help="train | val",
     default="train"},

    {name="split",
     type="number",
     help="Percentage of split to go to Training"},

}


function dataset:__init(...)

    -- argcheck
    local args =  initcheck(...)
    print(args)
    for k,v in pairs(args) do self[k] = v end

    -- find 1000 common vocabularies and construct index table
    self.vocToPosTable, self.vocToIndTable = textToTable(self.vocPath)
    self.classIndices = self.vocToIndTable

    ----------------------------------------------------------------------
    -- construct necessary file paths for obtaining image name and label
    local captionPath = paths.concat(
        self.path,
        'captions_' .. self.protocol .. '2014.tds.t7'
        )
    local imgDir = paths.concat(
        self.path,
        'images/' .. self.protocol .. '2014'
        )
    -- get id of sampled images, which will be placed in orders
    local smpId = nil
    if self.imageIdPath then
        smpId = {}
        local imgIdFile = io.open(self.imageIdPath, 'r')
        local str = imgIdFile:read("*all")
        imgIdFile:close()
        local idList = str:split('\n')
        for ind=1,#idList do
            local id = tonumber(idList[ind])
            smpId[id] = ind
        end
    end

    -- get image paths and their visual labels, visual concepts
    -- self.imagePath : path to each image in dataset
    -- self.imageClass : visual labels and concepts of each image
    self.imagePath, self.imageClass, self.imageClassCount,
        self.capClass, self.imgToCapInds = getMSCPathLab(
            captionPath,
            imgDir,
            self.classIndices,
            smpId
            )

    self.classes = {}
    for voc, ind in pairs(self.classIndices) do
        self.classes[ind] = voc
    end
    self.imgList = torch.LongTensor() -- indices of each image

    --==========================================================================

    self.numSamples = self.imagePath:size(1)
    print(self.numSamples ..  ' samples found.')
    self.imgList = torch.range(1, self.numSamples):long() -- fill image indices
    self.imgListSample = self.imgList -- list used in dataset:sample()

    --==========================================================================

    if self.split == 100 then
        self.testIndicesSize = 0
    else
        print('Splitting training and test sets to a ratio of '
              .. self.split .. '/' .. (100-self.split))
        -- split the imgList into imgListTrain and imgListTest
        local numTest = math.ceil((100 - self.split) / 100 * self.numSamples)
        if numTest >= self.numSamples then
            self.testIndicesSize = self.numSamples
        else
            local order = torch.randperm(self.numSamples):long()
            self.imgListTest = order[{{1, numTest}}]:clone()
            self.imgListTrain = order[{{numTest+1, -1}}]:clone()
            self.imgListSample = self.imgListTrain

            -- Now combine classListTest into a single tensor
            self.testIndices = self.imgListTest
            self.testIndicesSize = self.testIndices:nElement()
        end
    end
end

-- add visual feature and mil-prob to dataloader
function dataset:addCache(vfProbCache, method)
    -- Args:
    --   vfProbCache: {vf Tensor, mil-prob Tensort}
    --   method: cat | sub
    assert(torch.type(vfProbCache) == 'table',
           'Unpermitted data format of vfProbCache, should be table instaed...')
    assert(#vfProbCache == 2, 'Missing data...')

    self.imageFeature = vfProbCache[1]
    if method == 'sub' then
        -- Substitute caption labels with pre-trained predictions
        self.capClass = vfProbCache[2]
        for imgInd, capInds in ipairs(self.imgToCapInds) do
            capInds = {imgInd}
            self.imgToCapInds[imgInd] = capInds
        end
    elseif method == 'cat' then
        -- COncat captino labels with pre-trained predictions
        local origNumCapClass = self.capClass:size(1)
        local addNumCapClass = vfProbCache[2]:size(1)
        self.capClass = torch.cat({self.capClass, vfProbCache[2]}, 1)
        for imgInd, capInds in ipairs(self.imgToCapInds) do
            capInds[#capInds+1] = imgInd + origNumCapClass
            self.imgToCapInds[imgInd] = capInds
        end
    elseif method ~= 'orig' then
        error('Unknown method of processing input visual concept predictions')
    end
end

-- compute number of positive/negative samples in every labels
function dataset:posStat()
    local stat = torch.sum(self.imageClass, 1)
    return stat
end

-- size()
function dataset:size()
    return self.imgList:size(1)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, tab)
    local tensor
    local quantity = #tab
    local iSize = tab[1]:size():totable()
    local tSize = {quantity}
    for _, dim in ipairs(iSize) do table.insert(tSize, dim) end
    tensor = torch.Tensor(table.unpack(tSize)):fill(-1)
    for i=1,quantity do
        tensor[i]:copy(tab[i])
    end
    return tensor
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
    assert(quantity)
    local dataTable = {}
    local scalarTable = {}
    local vfTable = {}
    for i=1,quantity do
        -- randomly sample
        local sampleIndex = torch.random(1, self.imgListSample:nElement())
        local index = self.imgListSample[sampleIndex]

        -- ground truth labels
        local class = self.imageClass[index]
        table.insert(scalarTable, class)

        -- pre-trained predictions
        local capInds = self.imgToCapInds[index]
        local capInd = capInds[torch.random(1, #capInds)]
        local capClass = self.capClass[capInd]
        local out = self:sampleHookTrain(capClass)
        table.insert(dataTable, out)

        -- pre-trained visual features
        local vf = self.imageFeature[index]
        table.insert(vfTable, vf)
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    local visualFeature = tableToOutput(self, vfTable)

    return data, scalarLabels, visualFeature
end

function dataset:genInputs(quantity)
    local data, scalarLabels, visualFeature = self:sample(quantity)
    local outputs = {data, visualFeature}
    local labels = {scalarLabels, visualFeature}

    return outputs, labels
end

function dataset:get(i1, i2)
    local indices = torch.range(i1, i2);
    local quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    local vfTable = {}
    for i=1,quantity do
        -- load the sample
        local index = indices[i]

        -- ground truth labels
        local class = self.imageClass[index]
        table.insert(scalarTable, class)

        -- pre-trained predictions
        local capInds = self.imgToCapInds[index]
        local capInd = capInds[#capInds]
        local capClass = self.capClass[capInd]
        local out = self:sampleHookTest(capClass)
        table.insert(dataTable, out)

        -- pre-trained visual features
        local vf = self.imageFeature[index]
        table.insert(vfTable, vf)
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    local visualFeature = tableToOutput(self, vfTable)

    return data, scalarLabels, visualFeature
end

function dataset:getInputs(i1, i2)
    local data, scalarLabels, visualFeature = self:get(i1, i2)
    local outputs = {data, visualFeature}
    local labels = {scalarLabels, visualFeature}

    return outputs, labels
end

return dataset
