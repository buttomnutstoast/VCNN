local noisyNOR, Parent = torch.class('nn.noisyNOR', 'nn.Module')

function noisyNOR:__init(threshold)
    Parent.__init(self)
    self.size = {}
    self.thr = threshold or 1
end

function noisyNOR:updateOutput(input)
    self.output:resizeAs(input):copy(input)
    -- P = 1 - (1-p1)(1-p2)..(1-pn)
    self.output:mul(-1):add(1)
    self.size = input:size():totable()
    self.output = self.output:view(self.size[1],self.size[2],-1):prod(3):squeeze(3)
    self.output:mul(-1):add(1)
    local max_pool = input:max(4):max(3)
    self.output:cmax(max_pool)
    return self.output
end

function noisyNOR:updateGradInput(input, gradOutput)
    local n,c,h,w = unpack(self.size)
    local gradTmp = self.output:clone()
    gradTmp:mul(-1):add(1)
    gradTmp = gradTmp:view(n,c,1,1):expand(n,c,h,w)

    local tmp = input:clone():mul(-1):add(1):cmax(1e-15)

    self.gradInput:resizeAs(gradTmp):copy(gradTmp)
    self.gradInput:cdiv(tmp)
    self.gradInput:maskedFill(torch.gt(self.gradInput, self.thr), self.thr)
    -- self.gradInput:maskedFill(torch.lt(self.gradInput, -self.eps),-self.eps)
    self.gradInput:cmul(gradOutput:view(n,c,1,1):expand(n,c,h,w))
    -- if torch.min(self.gradInput) < -1e-3 then
    --     self.gradInput:div(torch.abs(torch.min(self.gradInput))):mul(0.001)
    -- end
    return self.gradInput
end