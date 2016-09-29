local ViewAs, parent = torch.class('nn.ViewAs', 'nn.Module')

function ViewAs:__init()
   parent.__init(self)
   self.size = -1
   self.gradInput = {}
end

function ViewAs:updateOutput(input)
   self.size = input[1]:size()
   self.output = input[1]:view(input[2]:size())
   return self.output
end

function ViewAs:updateGradInput(input, gradOutput)
   self.gradInput[1] = gradOutput:view(self.size)
   self.gradInput[2] = input[2]:clone():zero()
   return self.gradInput
end