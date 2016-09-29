-- Borrow code from Soumith Chintala
-- Note:
--   If you encountered 'Not enough memory' error,
--   please use Lua 5.x instead of LuaJIT 2.x
--   (LuaJIT limits memory usage up to 2GB)

local function parse(args)
   cmd = torch.CmdLine()
   cmd:text('Helper scripts to save mscoco json file to tds')
   cmd:option('-split', 'train', 'Options: train | val')
   cmd:option('-data',  'data/MSCOCO', '/path/to/MSCOCO/dir')

   local opt = cmd:parse(args or {})
    return opt
end

local opt = parse(arg)
assert(opt.split == 'train' or opt.split == 'val',
   'Unidentified option -split ' .. opt.split)

local jsonFile = string.format('annotations/captions_%s2014.json', opt.split)
local tbFile = string.format('annotations/captions_%s2014.table.t7', opt.split)
local tdsFile = string.format('captions_%s2014.tds.t7', opt.split)


if paths.filep(paths.concat(opt.data, tdsFile)) then
   print(string.format('File already saved at %s! Nothing to do...', tdsFile))
   os.exit(0)
else
   local json = require 'dkjson'
   local f = io.open(paths.concat(opt.data, jsonFile))
   local str = f:read('*all')
   f:close()

   m = json.decode(str)
   torch.save(paths.concat(opt.data, tbFile), m)
   print(string.format('Saved %s', paths.concat(opt.data, tbFile)))
end

m = torch.load(paths.concat(opt.data, tbFile))
local tds = require 'tds'

function recursiveTypeTableToTDS(m)
   local tp = torch.type(m)
   if tp == 'table' then
      local out = tds.hash()
      for k,v in pairs(m) do
     out[k] = recursiveTypeTableToTDS(v)
      end
      return out;
   elseif tp == 'string' or tp == 'number' then
      return m
   else
      error('unhandled type:', tp)
   end
end

m = recursiveTypeTableToTDS(m)
torch.save(paths.concat(opt.data, tdsFile), m)
print(string.format('Saved %s', paths.concat(opt.data, tdsFile)))
print('Finished!!')
collectgarbage();
