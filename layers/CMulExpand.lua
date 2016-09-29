local CMulExpand, parent = torch.class('nn.CMulExpand', 'nn.Module')

function CMulExpand:__init(...)
    -- This is a revised version of nn.CMul
    -- Initialized this modules as nn.CMul, but set size of selected dimensions
    -- to 1 where weights are shared.
    -- For example, set size (1,3,1) indicates weights are shared across dimension
    -- 1 and 3
    parent.__init(self)

    local arg = {...}

    self.size = torch.LongStorage()
    local n = #arg
    if n == 1 and torch.type(arg[1]) == 'torch.LongStorage' then
        self.size:resize(#arg[1]):copy(arg[1])
    else
        self.size:resize(n)
        for i=1,n do
          self.size[i] = arg[i]
        end
    end

    self.weight = torch.Tensor(self.size)
    self.gradWeight = torch.Tensor(self.size)

    self.output:resize(self.size)

    self:reset()
end

function CMulExpand:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1./math.sqrt(self.weight:nElement())
    end
    self.weight:uniform(-stdv,stdv)
end

function CMulExpand:updateOutput(input)
    self._weight = self._weight or input.new()

    self.output:resizeAs(input):copy(input)
    self._weight:expandAs(self.weight, input)
    self._expand_ratio = torch.cdiv(torch.Tensor(self._weight:size():totable()),
                                    torch.Tensor(self.weight:size():totable()))
    self.output:cmul(self._weight)
    return self.output
end

function CMulExpand:updateGradInput(input, gradOutput)
    self.gradInput = torch.cmul(self._weight, gradOutput)
    return gradInput
end

function CMulExpand:accGradParameters(input, gradOutput, scale)
    self._gradWeight = self._gradWeight or input.new()
    self._gradWeight:resizeAs(self._weight):zero()
    self._gradWeight:addcmul(scale, input, gradOutput)

    -- sum gradWeight over the dimension where tensor is expanded
    for ind=self._expand_ratio:nElement(),1,-1 do
        if self._expand_ratio[ind] > 1 then
            self._gradWeight = self._gradWeight:sum(ind)
        end
    end
    self.gradWeight:copy(self._gradWeight)
end

function CMulExpand:type(type, tensorCache)
   if type then
      self:clearState()
   end
   return parent.type(self, type, tensorCache)
end

function CMulExpand:clearState()
   nn.utils.clear(self, {
      '_input',
      '_output',
      '_weight',
      '_gradWeight',
      '_expand',
      '_repeat',
      '_sum',
   })
   return parent.clearState(self)
end