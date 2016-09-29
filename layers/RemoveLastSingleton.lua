local RemoveLastSingleton, Parent = torch.class('nn.RemoveLastSingleton', 'nn.Module')

function RemoveLastSingleton:__init()
    Parent.__init(self)
    self.size = -1
    self.didRemove = false
end

function RemoveLastSingleton:updateOutput(input)
    self.size = input:size():totable()
    if self.size[#self.size] == 1 then
        self.size[#self.size] = nil
        self.didRemove = true
    end
    self.output = input:view(table.unpack(self.size))
    return self.output
end

function RemoveLastSingleton:updateGradInput(input, gradOutput)
    if self.didRemove then
        self.size[#self.size+1] = 1
    end
    self.gradInput = gradOutput:view(table.unpack(self.size))
    self.didRemove = false
    return self.gradInput
end