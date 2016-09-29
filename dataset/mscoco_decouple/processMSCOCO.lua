--
-- Codes for extracting image file names and its ground-truth labels.
-- The definition of ground-truth labels is based on intersection of 1000 common
-- caption labels and 80 detection labels in MSCOCO. Manual mapping is adopted
-- to link captions with detection labels, and only 73 of 80 detection labels
-- used.
--

require 'paths'
require 'torch'
require 'tds'

local ffi = require 'ffi'
torch.setdefaulttensortype('torch.FloatTensor')

local function aTob(table, aDomain, bDomain)
    -- Helper to get new table mapping a to b
    --
    newtable = {}
    for ind = 1, #table do
        aD = table[ind][aDomain]
        bD = table[ind][bDomain]
        newtable[aD] = bD
    end
    return newtable
end

local function listFind(key, list)
    -- Helper to find if key in the list
    --
    if key == nil then
        return 0
    end

    for ind= 1, #list do
        if key == list[ind] then
            return ind
        end
    end
    return 0
end

local function repZeroNTimes(n)
    -- Helper to return 1*n tensor filled with zeros
    zeros = torch.FloatTensor(n):fill(0)
    return zeros
end

function textToTable(filePath)
    -- Helper to split (vocabulary, POS-tags, num_words) pairs into two tables
    -- One table is filled with (vocabulary : POS-tags) pairs
    -- Another is composed of (vocabulary : index) paris
    assert(paths.filep(filePath), filePath .. ' not exists!')
    -- open the file
    file = io.open(filePath, 'r')
    str = file:read("*all")
    file:close()

    -- replace ', ' with ','
    strList = str:split('\n')
    vocToPosTable = {}
    vocToIndTable = {}
    for ind, row in ipairs(strList) do
        rowList = row:split(", ")

        assert(#rowList == 3)
        voc, pos, num_words = rowList[1], rowList[2], rowList[3]

        -- insert (voc : pos) pair to vocToPosTable
        vocToPosTable[voc] = pos

        -- insert (voc : ind) pair to vocToIndTable
        vocToIndTable[voc] = ind
    end
    return vocToPosTable, vocToIndTable
end

function getMSCPathLab(capPath, imgDir, catInds, smpId)
    -- Retrive visual concepts from mscoco
    -- Args:
    --     capPath : path to caption.json
    --     imgDir  : root directory of MSCOCO images
    --     catInds : indices of specified category ID of detection labels
    --     smpId   : specified image id for sampling
    --
    -- Returns:
    --     image file paths, visual concepts and counts
    --
    assert(imgDir ~= '', 'imgDir should not be empty...')

    -- get caption annotations from mscoco api
    local captions = torch.load(capPath)
    -- Transform class name to class index
    local vocToInd = catInds
    local numVoc = 0
    for _, _ in pairs(vocToInd) do numVoc = numVoc + 1 end

    -- Retrieve image Id for sampling. If no ID provided, we principally use
    -- all images in the dataset.
    local smpImgId = {}
    if smpId == nil then
        local images = captions.images
        for ind=1,#images do
            id = images[ind].id
            smpImgId[id] = ind
        end
    else
        smpImgId = smpId
    end

    -- place images in order
    local imgIdOrder = {}
    local idCount = 0
    for id, ind in pairs(smpImgId) do
        imgIdOrder[ind] = id
        idCount = idCount + 1
    end
    assert(#imgIdOrder == idCount,
           'indices in (id : ind) pairs of smpId is not continuous!!')


    -- Construct table imgIdLabels to record image ID and its visual labels,
    -- which should be formatted as
    -- {
    --   id: {0, 1, 1,....},
    -- }
    local imgIdLabels = {}
    local imgIdPaths = {}
    local imgIdLabelCounts = {}
    local maxPathLength = 0

    -- Construct table capIndLabels to record visual labels from every caption
    -- which should be formatted as
    -- {
    --   id: {0, 1, 1,....},
    -- }
    local capIdLabels  = {}
    local imgIdToCapId = {}

    -- Retrieve detection labels
    local allCaps = captions.annotations
    local imgIdToName = aTob(captions.images, 'id', 'file_name')
    for capInd, capInfo in pairs(allCaps) do
        local imgId = capInfo.image_id -- Retrieve imageId of the caption
        local capId = capInfo.id -- Retrieve id of the caption
        -- Check if image id in the specified lists
        if smpImgId[imgId] then
            local caption = capInfo.caption:lower()
            local sub_caption = caption:gsub('[%.,]', '')
            local vocabs = sub_caption:split(' ')
            for ind=1,#vocabs do
                local voc = vocabs[ind]
                local vocInd = vocToInd[voc]
                if vocInd then
                    -- Insert imgId if not exists in the table
                    if imgIdLabels[imgId] == nil then
                        imgIdLabels[imgId] = repZeroNTimes(numVoc)
                        imgIdLabelCounts[imgId] = repZeroNTimes(numVoc)
                        local imgName = paths.concat(imgDir, imgIdToName[imgId])

                        if #imgName > maxPathLength then
                            maxPathLength = #imgName
                        end
                        imgIdPaths[imgId] = imgName
                    end
                    imgIdLabels[imgId][vocInd] = 1
                    local count = imgIdLabelCounts[imgId][vocInd]
                    imgIdLabelCounts[imgId][vocInd] = count + 1

                    -- Insert capId if not exists in the table
                    if capIdLabels[capId] == nil then
                        capIdLabels[capId] = repZeroNTimes(numVoc)
                        imgIdToCapId[imgId] = imgIdToCapId[imgId] or {}
                        table.insert(imgIdToCapId[imgId], capId)
                    end
                    capIdLabels[capId][vocInd] = 1
                end
            end
        end
    end

    -- Check if instance and caption file valid
    assert(maxPathLength > 0, "paths of files are length 0?")
    assert(#imgIdOrder > 0, "Could not find any image file in the given input paths")

    -- Obtain image name
    imgPaths = torch.CharTensor()
    imgLabels = torch.FloatTensor()
    imgLabelCounts = torch.FloatTensor()
    capLabels = torch.FloatTensor()
    -- initialize imgPaths and imgLabels
    local numImg = #imgIdOrder
    maxPathLength = maxPathLength + 1 -- add EOL to cstring
    imgPaths:resize(numImg, maxPathLength):fill(0)
    imgLabels:resize(numImg, numVoc):fill(0) -- visual concepts
    imgLabelCounts:resize(imgLabels:size()):fill(0) -- visual concepts count

    -- initialize capLabels
    local numCap = 0
    for _, _ in pairs(capIdLabels) do numCap = numCap + 1 end
    capLabels:resize(numCap, numVoc):fill(0)

    -- Read in image paths
    local s_data = imgPaths:data()
    local capInd = 0
    local imgIndToCapInds = {}
    for ind = 1, numImg  do
        local id = imgIdOrder[ind]
        local labels = imgIdLabels[id]
        local line = imgIdPaths[id]
        local counts = imgIdLabelCounts[id]
        -- load imgName into tensor
        ffi.copy(s_data, line)
        s_data = s_data + maxPathLength
        -- load imgLabels into tensor
        imgLabels[ind] = labels
        -- load imgLabelCounts into tensor
        imgLabelCounts[ind] = counts
        -- load capLabels into tensor
        capIds = imgIdToCapId[id]
        for _, capId in ipairs(capIds) do
            capInd = capInd + 1
            capLabels[capInd] = capIdLabels[capId]
            imgIndToCapInds[ind] = imgIndToCapInds[ind] or {}
            table.insert(imgIndToCapInds[ind], capInd)
        end
    end

    return imgPaths, imgLabels, imgLabelCounts, capLabels, imgIndToCapInds
end