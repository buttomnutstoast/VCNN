local ConstantAdd, Parent = torch.class('nn.ConstantAdd', 'nn.Module')

function ConstantAdd:__init(constVal,ValMuliplyToInput)
   Parent.__init(self)
   self.constVal = constVal or 1
   self.mv = ValMuliplyToInput or 1

   if self.mv == 0 then
      error('<ConstantAdd> Input should not be multiplied by zero.')
   end
end

function ConstantAdd:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:mul(self.mv):add(self.constVal)
   return self.output
end

function ConstantAdd:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:mul(self.mv)
   return self.gradInput
end