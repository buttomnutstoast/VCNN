-- GaussianSampler
-- samples a random vector from a given {mean,std}
--
-- Note: mean:s :size() is the same as mean:size()

local GaussianSampler, Parent = torch.class('nn.GaussianSampler', 'nn.Module')

function GaussianSampler:__init()
    Parent.__init(self)
    self.gradInput = {}
    self.randnOut = nil
end

function GaussianSampler:updateOutput(input)
    -- self.randnOut = input[1]:clone():copy(torch.randn(input[1]:size()))
    -- self.output:resizeAs(input[1]):copy(self.randnOut)
    -- self.output:cmul(input[2]):add(input[1])

    self.randnOut = self.randnOut or input[1].new()
    self.randnOut:resizeAs(input[1]):copy(torch.randn(input[1]:size()))

    self.ouput = self.output or self.output.new()
    self.output:resizeAs(input[2]):copy(input[2])
    self.output:mul(0.5):exp():cmul(self.randnOut)

    self.output:add(input[1])
    return self.output
end

function GaussianSampler:updateGradInput(input, gradOutput)
    -- self.gradInput[1] = self.gradInput[1] or gradOutput:clone()
    -- self.gradInput[2] = self.gradInput[2] or gradOutput:clone()

    -- self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)
    -- self.gradInput[2]:resizeAs(input[2]):copy(self.randnOut):cmul(gradOutput)
    -- self.randnOut = nil
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)

    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(gradOutput):copy(input[2])

    self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.randnOut)
    self.gradInput[2]:cmul(gradOutput)
    return self.gradInput
end