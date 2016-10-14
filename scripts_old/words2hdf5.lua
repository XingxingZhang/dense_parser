
require 'hdf5'
require 'torch'
include '../utils/shortcut.lua'

local SENT_END = {'###eos###'}

local function createVocab(inputFile, freqCut, ignoreCase, keepFreq)
  local fin = io.open(inputFile, 'r')
  local wordVector = {}
  local wordFreq = {}
  while true do
    local line = fin:read()
    if line == nil then break end
    local words = line:splitc(' \t\r\n')
    for _, word in ipairs(words) do
      if ignoreCase then word = word:lower() end
      if wordFreq[word] then
        wordFreq[word] = wordFreq[word] + 1
      else
        wordFreq[word] = 1
        wordVector[#wordVector + 1] = word
      end
    end
  end
  fin:close()
  
  local wid = 1
  local word2idx = {}
  if not wordFreq['UNK'] then
    word2idx = {UNK = wid}
    wid = wid + 1
  end
  local uniqUNK = 0
  local freqs = { 0 }
  for _, wd in ipairs(wordVector) do
    if wordFreq[wd] > freqCut then
      word2idx[wd] = wid
      freqs[wid] = wordFreq[wd]
      wid = wid + 1
    else
      uniqUNK = uniqUNK + 1
      if not wordFreq['UNK'] then
        freqs[1] = freqs[1] + wordFreq[wd]
      end
    end
  end
  word2idx[SENT_END[1]] = wid
  local vocabSize = wid
  -- wid = wid + 1
  
  local idx2word = {}
  for wd, i in pairs(word2idx) do
    idx2word[i] = wd
  end
  
  local vocab = {word2idx = word2idx, idx2word = idx2word, 
    freqCut = freqCut, ignoreCase = ignoreCase, 
    keepFreq = keepFreq, UNK = word2idx['UNK'],
    EOS = word2idx['###eos###']}
  if keepFreq then
    vocab['freqs'] = freqs
    vocab['uniqUNK'] = uniqUNK
    printf('freqs size %d\n', #freqs)
  end
  
  assert(vocabSize == table.len(word2idx))
  printf('original #words %d, after cut = %d, #words %d\n', #wordVector, freqCut, vocabSize)
  vocab['nvocab'] = vocabSize
  -- print(table.keys(vocab))
  for k, v in pairs(vocab) do
    printf('%s -- ', k)
    if type(v) ~= 'table' then
      print(v)
    else
      print('table')
    end
  end
  
  return vocab
end

local function words2hdf5(vocab, h5out, splitLabel, splitFile, maxlen)
  local gdata = string.format('/%s/x_data', splitLabel)
  local gindex = string.format('/%s/index', splitLabel)
  local fin = io.open(splitFile, 'r')
  local word2idx = vocab.word2idx
  
  local dOpt = hdf5.DataSetOptions()
  dOpt:setChunked(1024*50*10)
  -- dOpt:setDeflate()
  local iOpt = hdf5.DataSetOptions()
  iOpt:setChunked(1024*10, 2)
  -- iOpt:setDeflate()
  
  local lineNo, offset = 0, 1
  local x_data = {}
  local index = {}
  local isFirst = true
  
  function appendData()
    local data_ = torch.IntTensor(x_data)
    local ind_ = torch.IntTensor(index)
    if not isFirst then
      h5out:append(gdata, data_, dOpt)
      h5out:append(gindex, ind_, iOpt)
    else
      h5out:write(gdata, data_, dOpt)
      h5out:write(gindex, ind_, iOpt)
      isFirst = false
    end
  end
  
  local ndel = 0
  
  while true do
    local line = fin:read()
    if line == nil then break end
    local words = line:splitc(' \t\r\n')
    -- print(#words, maxlen)
    if #words <= maxlen then
      local xs = {}
      local idx = {}
      for _, word in ipairs(words) do
        word = vocab.ignoreCase and word:lower() or word
        local wid = word2idx[word] or vocab.UNK
        xs[#xs + 1] = wid
      end
      local vlen = #xs
      idx = {offset, vlen}
      table.extend(x_data, xs)
      index[#index + 1] = idx
      
      offset = offset + vlen
      lineNo = lineNo + 1
      if lineNo % 50000 == 0 then
        appendData()
        x_data = {}
        index = {}
        collectgarbage()
      end
    else
      ndel = ndel + 1
    end
  end
  
  if #x_data > 0 then
    appendData()
  end
  
  printf('[%s] delete %d sentences\n', splitLabel, ndel)
  
  fin:close()
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('== convert text to hdf5 format ==')
  cmd:text()
  cmd:option('--train', '', 'train text file')
  cmd:option('--valid', '', 'valid text file')
  cmd:option('--test', '', 'test text file')
  cmd:option('--dataset', '', 'the resulting dataset (.h5)')
  cmd:option('--freq', 0, 'words less than or equals to \"freq\" times will be ignored')
  cmd:option('--ignorecase', false, 'case will be ignored')
  cmd:option('--keepfreq', false, 'keep frequency information during creating vocabulary')
  cmd:option('--maxlen', 100, 'sentences longer than maxlen will be ignored!')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  print(opts)
  local vocab = createVocab(opts.train, opts.freq, opts.ignorecase, opts.keepfreq)
  assert(opts.dataset:ends('.h5'), 'dataset must be hdf5 file .h5')
  local dataPrefix = opts.dataset:sub(1, -4)
  local vocabPath = dataPrefix .. '.vocab.t7'
  printf('save vocab to %s\n', vocabPath)
  torch.save(dataPrefix .. '.vocab.t7', vocab)
  
  local h5out = hdf5.open(opts.dataset, 'w')
  words2hdf5(vocab, h5out, 'train', opts.train, opts.maxlen)
  print('create training set done!')
  words2hdf5(vocab, h5out, 'valid', opts.valid, opts.maxlen)
  print('create validing set done!')
  words2hdf5(vocab, h5out, 'test', opts.test, opts.maxlen)
  print('create testing set done!')
  h5out:close()
  printf('save dataset to %s\n', opts.dataset)
end

main()
