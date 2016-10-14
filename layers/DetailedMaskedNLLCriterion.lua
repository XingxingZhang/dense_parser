
--[[
-- when y contains zeros
--]]

local DetailedMaskedNLLCriterion, parent = torch.class('DetailedMaskedNLLCriterion', 'nn.Module')

function DetailedMaskedNLLCriterion:__init()
  parent.__init(self)
end

function DetailedMaskedNLLCriterion:updateOutput(input_)
  local input, target, div = unpack(input_)
  div = div or 1
  if input:dim() == 2 then
    self.output:resize(target:size())
    self.output:zero()
    local nlls = self.output
    local n = target:size(1)
    for i = 1, n do
      if target[i] ~= 0 then
        nlls[i] = -input[i][target[i]]
      else
        nlls[i] = 0
      end
    end
    
    return self.output
  else
    error('input must be matrix! Note only batch mode is supported!')
  end
end

function DetailedMaskedNLLCriterion:updateGradInput(input_)
  local input, target, div = unpack(input_)
  div = div or 1
  
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  local er = -1 / div
  if input:dim() == 2 then
    local n = target:size(1)
    for i = 1, n do
      if target[i] ~= 0 then
        self.gradInput[i][target[i]] = er
      end
    end
    return self.gradInput
  else
    error('input must be matrix! Note only batch mode is supported!')
  end
end

