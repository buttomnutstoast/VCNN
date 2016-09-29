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
    self.imagePath, self.imageClass, self.imageClassCount, _, _ = getMSCPathLab(
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

function dataset:get(i1, i2)
    local indices = torch.range(i1, i2);
    local quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        -- load the sample
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
        local out = self:imgHook(imgpath)
        table.insert(dataTable, out)
        table.insert(scalarTable, self.imageClass[indices[i]])
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:getInputs(i1, i2)
    local data, scalarLabels = self:get(i1, i2)
    return {data}, {scalarLabels}
end

return dataset
