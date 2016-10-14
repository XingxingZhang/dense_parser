
-- directly copied from ff-lstm-v1.6

require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'EMaskedClassNLLCriterion'

require 'basic'

local model_utils = require 'model_utils'
local lstm_util = require 'lstmutil'

local FFLSTMLM = torch.class('FFLSTMLM', 'BModel')

function FFLSTMLM:__init(opts)
  self.opts = opts
  self.name = 'FFLSTMLM'
  self:print( 'build FFLSTMLM ...' )
  opts.nivocab = opts.nivocab or opts.nvocab
  opts.novocab = opts.novocab or opts.nvocab
  
  self:createNetwork()
  
  if opts.optimMethod == 'AdaGrad' then
    self.optimMethod = optim.adagrad
  elseif opts.optimMethod == 'Adam' then
    self.optimMethod = optim.adam
  elseif opts.optimMethod == 'AdaDelta' then
    self.optimMethod = optim.adadelta
  elseif opts.optimMethod == 'SGD' then
    self.optimMethod = optim.sgd
  end
  
  self:print( 'build FFLSTMLM done!' )
end

function FFLSTMLM:transData(d)
  if self.opts.useGPU then 
    return d:cuda() 
  else
    return d
  end
end

function FFLSTMLM:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
  -- compute activations of four gates all together
  local x2h = nn.Linear(nin, nhid * 4)(x_t)
  local h2h = nn.Linear(nhid, nhid * 4)(h_tm1)
  local allGatesActs = nn.CAddTable()({x2h, h2h})
  local allGatesActsSplits = nn.SplitTable(2)( nn.Reshape(4, nhid)(allGatesActs) )
  -- unpack all gate activations
  local i_t = nn.Sigmoid()( nn.SelectTable(1)( allGatesActsSplits ) )
  local f_t = nn.Sigmoid()( nn.SelectTable(2)( allGatesActsSplits ) )
  local o_t = nn.Sigmoid()( nn.SelectTable(3)( allGatesActsSplits ) )
  local n_t = nn.Tanh()( nn.SelectTable(4)( allGatesActsSplits ) )
  -- compute new cell
  local c_t = nn.CAddTable()({
      nn.CMulTable()({ i_t, n_t }),
      nn.CMulTable()({ f_t, c_tm1 })
    })
  -- compute new hidden state
  local h_t = nn.CMulTable()({ o_t, nn.Tanh()( c_t ) })
  
  return c_t, h_t
end

function FFLSTMLM:createGRU(x_t, h_tm1, nin, nhid)
  -- computes reset gate and update gate together
  local x2h = nn.Linear(nin, nhid * 2)(x_t)
  local h2h = nn.Linear(nhid, nhid * 2)(h_tm1)
  local allGatesActs = nn.CAddTable()({x2h, h2h})
  local allGatesActsSplits = nn.SplitTable(2)( nn.Reshape(2, nhid)(allGatesActs) )
  local r_t = nn.Sigmoid()( nn.SelectTable(1)( allGatesActsSplits ) )
  local z_t = nn.Sigmoid()( nn.SelectTable(2)( allGatesActsSplits ) )
  
  local h_hat_t = nn.Tanh()(
    nn.CAddTable()({ 
        nn.Linear(nin, nhid)(x_t),
        nn.Linear(nhid, nhid)(
          nn.CMulTable()({r_t, h_tm1})
          ) 
      })
    )
  
  local h_t = nn.CAddTable()({
      nn.CMulTable()({ z_t, h_hat_t }),
      nn.CMulTable()({ nn.AddConstant(1)(nn.MulConstant(-1)(z_t)), h_tm1})
    })
  
  return h_t
end

function FFLSTMLM:createDeepFFLSTM(opts)
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  -- this is for the n-gram model
  local nx_t = nn.Identity()()
  -- this is for the combined hidden state
  -- do I actually need this?
  -- local h_hat_tm1 = nn.Identity()()
  
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  -- this is the computation for LSTM
  local in_t = { [0] = emb(x_t):annotate{name='lstm_lookup'} }
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
    
    if opts.dropout > 0 then
      printf( 'lstm encoder layer %d, dropout = %f\n', i, opts.dropout) 
      x_t_i = nn.Dropout(opts.dropout)(x_t_i)
    end
    
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nin, opts.nhid)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  local h_L = s_t[2*opts.nlayers]
  -- done with the computation of LSTM
  
  -- let's start the NNLM computation; currently there is only one layer in NNLM
  local emb2 = nn.LookupTable(opts.nivocab, opts.nin)
  local ngram_repr_ = nn.Linear( opts.window * opts.nin, opts.nhid )(
    nn.View( opts.window * opts.nin )( emb2(nx_t):annotate{name='ngram_lookup'} )
    )
  
  -- this is for one-layer case
  --[[
  local ngram_repr
  if opts.nnactiv == 'tanh' then
    ngram_repr = nn.Tanh()( ngram_repr_ )
    print 'NN activation function: tanh'
  elseif opts.nnactiv == 'relu' then
    ngram_repr = nn.ReLU()( ngram_repr_ )
    print 'NN activation function: relu'
  else
    error('not recognized nnactiv')
  end
  --]]
  
  local ngram_repr
  if opts.nnactiv == 'tanh' then
    ngram_repr = nn.Tanh()( ngram_repr_ )
    print 'NN activation function: tanh'
  elseif opts.nnactiv == 'relu' then
    ngram_repr = nn.ReLU()( ngram_repr_ )
    print 'NN activation function: relu'
  else
    error('not recognized nnactiv')
  end
  
  for i = 1, opts.nlayers - 1 do
    if opts.dropout > 0 then
      ngram_repr = nn.Dropout(opts.dropout)(ngram_repr)
    end
    local ngram_lrepr = nn.Linear(opts.nhid, opts.nhid)(ngram_repr)
    if opts.nnactiv == 'tanh' then
      ngram_repr = nn.Tanh()( ngram_lrepr )
    elseif opts.nnactiv == 'relu' then
      ngram_repr = nn.ReLU()( ngram_lrepr )
    end
    printf('create NNLM layer %d done!\n', i + 1)
  end
  
  if opts.dropout > 0 then
    printf('NNLM is using dropout %f\n', opts.dropout)
    ngram_repr = nn.Dropout(opts.dropout)(ngram_repr)
  end
  
  local h_hat_t
  if opts.combine == 'GRU' then
    print('Using GRU for combination!')
    -- let's use a GRU to combine these two representations
    h_hat_t = self:createGRU(ngram_repr, h_L, opts.nhid, opts.nhid)
  elseif opts.combine == 'Linear' then
    print('Using Linear for combination')
    h_hat_t = nn.Tanh()( nn.Linear(2*opts.nhid, opts.nhid)( nn.JoinTable(2){h_L, ngram_repr} ) )
  end
  
  if opts.dropout > 0 then
    h_hat_t = nn.Dropout(opts.dropout)(h_hat_t)
    printf('apply dropout before output layer, drop = %f\n', opts.dropout)
  end
  
  local y_a = nn.Linear(opts.nhid, opts.novocab)(h_hat_t)
  local y_prob = nn.LogSoftMax()(y_a)
  
  local model = nn.gModule({x_t, nx_t, s_tm1}, {nn.Identity()(s_t), y_prob})
  
  return self:transData(model)
end

function FFLSTMLM:initModel(opts)
  self.fflstm_hs = {}   -- including all h_t and c_t
  for j = 0, opts.seqLen do
    self.fflstm_hs[j] = {}
    for d = 1, 2 * opts.nlayers do
      self.fflstm_hs[j][d] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    end
  end
  
  -- df of fflstm model
  self.df_fflstm_h = {}
  for i = 1, 2 * opts.nlayers do
    self.df_fflstm_h[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
  
  self.nx_ts = {}
  for i = 1, opts.seqLen do
    self.nx_ts[i] = self:transData( torch.zeros(opts.batchSize, opts.window):long() )
  end
end

function FFLSTMLM:createNetwork()
  self.fflstm_master = self:createDeepFFLSTM(self.opts)
  -- model_utils.show_nngraph_names( self.fflstm_master )
  if self.opts.shareEmbed then
    self:print('use share embedding matrices for both NNLM and LSTM')
    self.params, self.grads = model_utils.combine_fflstm_parameters(self.fflstm_master)
    self.params:uniform(-self.opts.initRange, self.opts.initRange)
    self:print( string.format( '#params %d', self.params:size(1) ) )
    model_utils.share_fflstm_lookup(self.fflstm_master)
  else
    self:print('use seperate embedding matrices for NNLM and LSTM')
    self.params, self.grads = model_utils.combine_all_parameters(self.fflstm_master)
    self.params:uniform(-self.opts.initRange, self.opts.initRange)
    self:print( string.format( '#params %d', self.params:size(1) ) )
  end
  
  self:print('clone fflstm ...')
  self.fflstms = model_utils.clone_many_times(self.fflstm_master, self.opts.seqLen)
  self:print('clone fflstm done!')
  
  self.criterions = {}
  for i = 1, self.opts.seqLen do
    self.criterions[i] = self:transData(EMaskedClassNLLCriterion())
  end
  
  self:initModel(self.opts)
end

function FFLSTMLM:trainBatch(x, y, sgdParam)
  assert(x[{ 1, {} }]:eq(self.opts.vocab.EOS):sum() == x:size(2), 'x should start with all EOS')
  x = self:transData(x)
  y = self:transData(y)
  
  self.fflstm_master:training()
  for i = 1, #self.fflstms do
    self.fflstms[i]:training()
  end
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    for i = 1, 2 * self.opts.nlayers do
      self.fflstm_hs[0][i]:zero()
    end
    
    -- this is for the forward pass
    local T = x:size(1)
    local loss = 0
    -- local model = nn.gModule({x_t, nx_t, s_tm1}, {nn.Identity()(s_t), y_prob})
    local y_preds = {}
    for t = 1, T do
      if t >= self.opts.window then
        self.nx_ts[t] = x[{ {t-self.opts.window+1, t}, {} }]:t()
      else
        self.nx_ts[t][{ {}, {1, self.opts.window-t} }] = self.opts.vocab.EOS
        self.nx_ts[t][{ {}, {self.opts.window-t+1, -1} }] = x[{ {1, t}, {} }]:t()
      end
      -- printf('nx_t size \n')
      -- print(self.nx_ts[t]:size())
      
      self.fflstm_hs[t], y_preds[t] = unpack( self.fflstms[t]:forward({
            x[{ t, {} }], self.nx_ts[t], self.fflstm_hs[t-1]
            }) )
      
      -- print(self.fflstm_hs[t])
      
      local loss_ = self.criterions[t]:forward({y_preds[t], y[{ t, {} }], self.opts.batchSize})
      loss = loss + loss_
      
      if self.opts.useGPU then cutorch.synchronize() end
      
      -- printf('forward t = %d\n', t)
    end
    
    -- printf('forward pass done! loss = %f\n', loss)
    
    -- this is for backward pass
    for i = 1, 2 * self.opts.nlayers do
      self.df_fflstm_h[i]:zero()
      -- printf('df_fflstm_h size')
      -- print(self.df_fflstm_h[i]:size())
    end
    -- print(self.df_fflstm_h)
    
    for t = T, 1, -1 do
      local df_crit = self.criterions[t]:backward({y_preds[t], y[{ t, {} }], self.opts.batchSize})
      -- print(df_crit:size())
      -- printf('t = %d\n', t)
      -- print(self.nx_ts[t]:size())
      local _, _, tmp =  unpack( self.fflstms[t]:backward(
        {x[{ t, {} }], self.nx_ts[t], self.fflstm_hs[t-1]},
        {self.df_fflstm_h, df_crit}
        ) )
      -- print('finish!')
      model_utils.copy_table(self.df_fflstm_h, tmp)
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    if self.opts.gradClip < 0 then
      local clip = -self.opts.gradClip
      self.grads:clamp(-clip, clip)
    elseif self.opts.gradClip > 0 then
      local maxGradNorm = self.opts.gradClip
      local gradNorm = self.grads:norm()
      if gradNorm > maxGradNorm then
        local shrinkFactor = maxGradNorm / gradNorm
        self.grads:mul(shrinkFactor)
      end
    end
    
    return loss, self.grads
  end
  
  local _, loss_ = self.optimMethod(feval, self.params, sgdParam)
  return loss_[1]
end


function FFLSTMLM:validBatch(x, y)
  assert(x[{ 1, {} }]:eq(self.opts.vocab.EOS):sum() == x:size(2), 'x should start with all EOS')
  x = self:transData(x)
  y = self:transData(y)
  
  self.fflstm_master:evaluate()
  for i = 1, #self.fflstms do
    self.fflstms[i]:evaluate()
  end
  
  for i = 1, 2 * self.opts.nlayers do
    self.fflstm_hs[0][i]:zero()
  end
  
  -- this is for the forward pass
  local T = x:size(1)
  local loss = 0
  -- local model = nn.gModule({x_t, nx_t, s_tm1}, {nn.Identity()(s_t), y_prob})
  local y_preds = {}
  for t = 1, T do
    if t >= self.opts.window then
      self.nx_ts[t] = x[{ {t-self.opts.window+1, t}, {} }]:t()
    else
      self.nx_ts[t][{ {}, {1, self.opts.window-t} }] = self.opts.vocab.EOS
      self.nx_ts[t][{ {}, {self.opts.window-t+1, -1} }] = x[{ {1, t}, {} }]:t()
    end
    
    self.fflstm_hs[t], y_preds[t] = unpack( self.fflstms[t]:forward({
          x[{ t, {} }], self.nx_ts[t], self.fflstm_hs[t-1]
          }) )
    local loss_ = self.criterions[t]:forward({y_preds[t], y[{ t, {} }], self.opts.batchSize})
    loss = loss + loss_
    
    if self.opts.useGPU then cutorch.synchronize() end
  end
  
  return loss, y_preds
end

