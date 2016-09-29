local ffi=require 'ffi'
require 'cunn'

function makeDataParallel(model, nGPU, net)
   if nGPU >= 1 then -- modified by Jerry: for others to train with multiple GPU
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1,true,true)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end

      -- allow multi-threads for multi-GPUS
      local option = opt
      local netobj = net
      local initFun = function()
         backend = option.backend
         packages = netobj.packages
         if packages then packages() end
         if backend == 'cudnn' then require 'cudnn' end
      end
      model:threads(initFun)
   end

   cutorch.setDevice(opt.GPU)
   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1, true, true)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU, net)
   -- load require packages
   if net.packages then net.packages() end
   if opt.backend == 'cudnn' then require 'cudnn' end

   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU, net)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU, net)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function removeParallelTb(filename)
    local model = torch.load(filename)
    local outModel = nn.Sequential()
    if torch.type(model) == 'nn.DataParallelTable' then
        outModel = model:get(1):float()
    elseif torch.type(model) == 'nn.Sequential' then
        for i,module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                outModel:add(module:get(1):float())
            else
                outModel:add(module:float())
            end
        end
    else
        error('The loaded model is not a Sequential or DataParallelTable module.')
    end
    return outModel
end