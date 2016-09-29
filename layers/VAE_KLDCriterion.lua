-- VAE_KLDCriterion
-- KLDivergence for VAE
--
-- out = 0.5 * sum(1+log(std^2)-mean^2-std^2)
--
-- inputs[1] = mean, inputs[2] = log(std^2), targets = ignore

local VAE_KLDCriterion, parent = torch.class('nn.VAE_KLDCriterion', 'nn.Criterion')

function VAE_KLDCriterion:updateOutput(inputs, targets)

    -- local mean_sq = torch.pow(inputs[1], 2)
    -- local std_sq = torch.pow(inputs[2], 2)
    -- local log_std_sq = torch.log(std_sq)
    -- log_std_sq:add(1):add(-mean_sq):add(-std_sq)
    -- self.output = -0.5 * torch.sum(log_std_sq)
    -- return self.output
    local mean, log_var = inputs[1], inputs[2]
    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()

    KLDelements:exp():mul(-1)
    KLDelements:add(-1, mean_sq)
    KLDelements:add(1)
    KLDelements:add(log_var)

    self.output = -0.5 * torch.sum(KLDelements) / KLDelements:nElement()

    return self.output
end

function VAE_KLDCriterion:updateGradInput(inputs, targets)
    self.gradInput = {}
    self.gradInput[1] = inputs[1]:clone():div(inputs[1]:nElement())

    -- self.gradInput[2] = torch.pow(inputs[2], 2)
    -- self.gradInput[2]:add(-1):cdiv(inputs[2])

    -- derivative of -1/2 * sum(log(std^2) - std^2) should be std - 1/std
    -- local std = inputs[2]:clone()
    -- self.gradInput[2] = std:add(-torch.cinv(std))
    self.gradInput[2] = torch.exp(inputs[2]):mul(-1):add(1):mul(-0.5):div(inputs[2]:nElement())
    return self.gradInput
end
