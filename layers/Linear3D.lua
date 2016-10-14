local Linear, parent = torch.class('Linear3D', 'nn.Module')

function Linear:__init(inputSize, outputSize, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self:reset()
end

function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

function Linear:updateOutput(input_)
  if input_:dim() == 3 then
    local input = input_:view(input_:size(1) * input_:size(2), input_:size(3))
    local nframe = input:size(1)
    local nElement = self.output:nElement()
    self.output:resize(nframe, self.weight:size(1))
    if self.output:nElement() ~= nElement then
       self.output:zero()
    end
    self.addBuffer = self.addBuffer or input.new()
    if self.addBuffer:nElement() ~= nframe then
       self.addBuffer:resize(nframe):fill(1)
    end
    self.output:addmm(0, self.output, 1, input, self.weight:t())
    if self.bias then self.output:addr(1, self.addBuffer, self.bias) end

    self.output = self.output:view(input_:size(1), input_:size(2), self.output:size(2))
  else
    error('input must be 3D tensor')
  end
  
  return self.output
end

function Linear:updateGradInput(input_, gradOutput_)
   if self.gradInput then
     assert(input_:dim() == 3, 'input_ must be 3D tensor')
     assert(gradOutput_:dim() == 3, 'gradOutput_ must be 3D tensor')
     local input = input_:view(input_:size(1)*input_:size(2), input_:size(3))
     local gradOutput = gradOutput_:view(gradOutput_:size(1) * gradOutput_:size(2), gradOutput_:size(3))

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      else
        error('input right now must be a matrix')
      end

      self.gradInput = self.gradInput:view( input_:size(1), input_:size(2), self.gradInput:size(2) )

      return self.gradInput
   end
end

function Linear:accGradParameters(input_, gradOutput_, scale)
   assert(input_:dim() == 3, 'input_ must be 3D tensor')
   assert(gradOutput_:dim() == 3, 'gradOutput_ must be 3D tensor')
   local input = input_:view(input_:size(1)*input_:size(2), input_:size(3))
   local gradOutput = gradOutput_:view(gradOutput_:size(1) * gradOutput_:size(2), gradOutput_:size(3))

   scale = scale or 1
   if input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   else
     error('input right now must be a matrix')
   end
end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters

function Linear:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function Linear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
