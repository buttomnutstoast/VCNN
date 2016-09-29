local AddSingletonDim, Parent = torch.class('nn.AddSingletonDim', 'nn.Module')

function AddSingletonDim:__init()
   Parent.__init(self)
   self.size = -1
end

function AddSingletonDim:updateOutput(input)
   self.size = input:size():totable()
   table.insert(self.size, 1)
   self.output = input:view(table.unpack(self.size))
   return self.output
end

function AddSingletonDim:updateGradInput(input, gradOutput)
   table.remove(self.size)
   self.gradInput = gradOutput:view(table.unpack(self.size))
   return self.gradInput
end