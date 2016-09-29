local ViewAdv, parent = torch.class('nn.ViewAdv', 'nn.Module')

function ViewAdv:__init(...)
   parent.__init(self)
   self.size = {...}
   self.input_size = {}
   self.performSize = {}
end

function ViewAdv:updateOutput(input)
   local nEle = input:nElement()
   self.input_size  = input:size():totable()
   self.performSize = input:size():totable()
   assert(#self.input_size==#self.size)

   local isOneMinusone = 0
   local minusOneLoc = 0
   local currentEle = 1

   for i=1,#self.size do
      if self.size[i] == 0 then
         self.performSize[i] = self.input_size[i]
         currentEle = currentEle*self.input_size[i]
      elseif self.size[i] == -1 then
         isOneMinusone = isOneMinusone+1
         minusOneLoc = i
         if (isOneMinusone>1) then
            error('Should have only one -1 in nn.ViewAdv arguments.')
         end
      elseif self.size[i] > 0 then
         self.performSize[i] = self.size[i]
         currentEle = currentEle*self.performSize[i]
      end
   end

   self.performSize[minusOneLoc] = nEle/currentEle
   self.output = input:view(table.unpack(self.performSize))
   return self.output
end

function ViewAdv:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:view(table.unpack(self.input_size))
   return self.gradInput
end