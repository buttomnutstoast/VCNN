paths.dofile('dataset.lua')
local currentDir = paths.dirname(paths.thisfile())

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache files (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7') -- training metadata
local testCache = paths.concat(opt.cache, 'testCache.t7')  -- testing metadata
local evalCache = paths.concat(opt.cache, 'evalCache.t7') -- eval metadata
local vfProbCache = paths.concat(opt.cache, 'vfProbCache.t7')  -- visual feature/mil-prob metadata

assert(paths.filep(vfProbCache),
       'visual feature and mil-prob cache not exist'
       .. ', run dataset/mscoco_cap_vc/vf_main.lua first!')
vfProb = torch.load(vfProbCache)

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.imageCrop, opt.imageCrop}


--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

--- function to process input data, including corruption and mean-subtraction
local trainHook = function(self, data)
    local out = data:clone()
    return out
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)
    assert(trainLoader.path == paths.concat(opt.data),
           'cached files dont have the same path as opt.data. '
           .. 'Remove your cached files at: '
           .. trainCache .. ' and rerun the program')
else
    print('Creating training metadata')
    trainLoader = dataLoader{
        path = paths.concat(opt.data),
        vocPath = paths.concat(currentDir, opt.vocabs),
        imageIdPath = paths.concat(currentDir, 'splits', 'train.ids'),
        protocol = 'train',
        split = 100,
        }
    torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook
trainLoader:addCache(vfProb.trainCache, 'sub')
collectgarbage()

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns visual concepts
--]]

--- function to perform mean-subtraction
local testHook = function(self, data)
    local out = data:clone()
    return out
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)
    assert(testLoader.path == paths.concat(opt.data),
           'cached files dont have the same path as opt.data.'
           .. 'Remove your cached files at: '
           .. testCache .. ' and rerun the program')
else
    print('Creating test metadata')
    testLoader = dataLoader{
        path = paths.concat(opt.data),
        vocPath = paths.concat(currentDir, opt.vocabs),
        imageIdPath = paths.concat(currentDir, 'splits', 'test.ids'),
        protocol = 'val',
        split = 0,
        }
    torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
testLoader:addCache(vfProb.testCache, 'sub')
collectgarbage()

-- End of test loader section
--------------------------------------------------------------------------------
--[[
   Section 3: Create a eval data loader (evalLoader),
   which can iterate over the eval set and returns visaul concepts
--]]

-- evalLoader adopts same function as testLoader to load the image
if paths.filep(evalCache) then
    print('Loading eval metadata from cache')
    evalLoader = torch.load(evalCache)
    assert(evalLoader.path == paths.concat(opt.data),
           'cached files dont have the same path as opt.data.'
           .. 'Remove your cached files at: '
           .. evalCache .. ' and rerun the program')
else
    print('Creating eval metadata')
    evalLoader = dataLoader{
        path = paths.concat(opt.data),
        vocPath = paths.concat(currentDir, opt.vocabs),
        imageIdPath = paths.concat(currentDir, 'splits', 'eval.ids'),
        protocol = 'val',
        split = 0,
        }
    torch.save(evalCache, evalLoader)
end
evalLoader.sampleHookTest = testHook
evalLoader:addCache(vfProb.evalCache, 'sub')
collectgarbage()