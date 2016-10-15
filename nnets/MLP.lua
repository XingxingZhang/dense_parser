
require 'basic'
require 'shortcut'
require 'optim'

local mlp = torch.class('MLP', 'BModel')

local function transferData(useGPU, data)
  if useGPU then
    return data:cuda()
  else
    return data
  end
end

function mlp:__init(opts)
  self.opts = opts
  self.model = nn.Sequential()
  local nhids = opts.snhids:splitc(',')
  opts.nhids = {}
  for _, snhid in ipairs(nhids) do
    table.insert(opts.nhids, tonumber(snhid))
  end
  
  self.model = nn.Sequential()
  
  if opts.inDropout > 0 then
    self.model:add(nn.Dropout(opts.inDropout))
  end
  
  local nlayers = #opts.nhids
  for i = 2, nlayers do
    self.model:add(nn.Linear(opts.nhids[i-1], opts.nhids[i]))
    if i ~= nlayers then
      if opts.batchNorm then
        self.model:add( nn.BatchNormalization(opts.nhids[i]) )
      end
      
      if opts.activ == 'tanh' then
        self.model:add(nn.Tanh())
      elseif opts.activ == 'relu' then
        self.model:add(nn.ReLU())
      else
        error(opts.activ .. ' not supported!')
      end
      
      if opts.dropout > 0 then
        self.model:add(nn.Dropout(opts.dropout))
      end
      
    end
  end
  self.model:add(nn.LogSoftMax())
  print(self.model)
  self.criterion = nn.ClassNLLCriterion()
  
  if opts.useGPU then
    print 'use GPU!'
    self.model = self.model:cuda()
    self.criterion = self.criterion:cuda()
  end
  
  self.params, self.grads = self.model:getParameters()
  printf('#param %d\n', self.params:size(1))
  
  -- self.model = require('weight-init')(self.model, 'kaiming')
  
  if opts.optimMethod == 'AdaGrad' then
    self.optimMethod = optim.adagrad
  elseif opts.optimMethod == 'SGD' then
    self.optimMethod = optim.sgd
  elseif opts.optimMethod == 'Adam' then
    self.optimMethod = optim.adam
  end
  
end

function mlp:trainBatch(x, y, sgd_params)
  self.model:training()
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
  end
  
  local feval = function(newParam)
    if self.params ~= newParam then
      self.params:copy(newParam)
    end
    
    self.grads:zero()
    local output = self.model:forward(x)
    local loss = self.criterion:forward(output, y)
    local df = self.criterion:backward(output, y)
    self.model:backward(x, df)

    return loss, self.grads
  end
  
  local _, loss_ = self.optimMethod(feval, self.params, sgd_params)
  
  return loss_[1]
end

function mlp:validBatch(x, y)
  if self.opts.useGPU then
    y = y:cuda()
  end
  
  local yPred = self:predictBatch(x)
  local maxv, maxi = yPred:max(2)
  return torch.sum( torch.eq(maxi, y) ), x:size(1), maxi
end

function mlp:predictBatch(x)
  self.model:evaluate()
  if self.opts.useGPU then
    x = x:cuda()
  end
  
  return torch.exp(self.model:forward(x))
end

