local TableCat, parent = torch.class('nn.TableCat', 'nn.Module')

function TableCat:__init()
   parent.__init(self)
   self.lens = {}

   self.output = {}
   self.gradInput = {}
end

function TableCat:updateOutput(input)
   self.output = {}
   self.lens = {}
   local n = 0

   for j=1,#input do
      for i=1, #input[j] do
         self.output[n+i] = input[j][i]
      end
      self.lens[j] = #input[j]
      n = n+#input[j]
   end
   return self.output
end

function TableCat:updateGradInput(input, gradOutput)
   local m = 0
   for j=1, #self.lens do
      for i=1, self.lens[j] do
         if self.gradInput[j] == nil then
            self.gradInput[j] = {}
         end
         self.gradInput[j][i] = gradOutput[m+i]
      end
      m = m+self.lens[j]
   end
   return self.gradInput
end