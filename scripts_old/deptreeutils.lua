
require 'torch'
include '../utils/shortcut.lua'
require 'depgraph'

local DepTreeUtils = torch.class('DepTreeUtils')

DepTreeUtils.ROOT_MARK = '###root###'
DepTreeUtils.UNK_TOKEN = '###unk###'

function DepTreeUtils.parseDepStr(depStr)
  local depWords = {}
  local pat = '([%w_]+)%((.-)%-(%d+), (.-)%-(%d+)%)'
  local mats = depStr:gmatch(pat)
  for rel, w1, p1, w2, p2 in mats do
    if w1 == 'ROOT' and p1 == '0' then
      w1 = DepTreeUtils.ROOT_MARK
    end
    -- w1 = w1 == 'ROOT' and DepTreeUtils.ROOT_MARK or w1
    table.insert(depWords, {rel=rel, w1 = w1, 
        p1 = tonumber(p1), w2 = w2, p2 = tonumber(p2)})
  end
  return depWords
end

function DepTreeUtils.dwords2words(dwords)
  local words = {}
  local maxID, cnt = -1, 0
  for _, dword in ipairs(dwords) do
    local wid = dword.p2 + 1
    words[wid] = dword.w2
    if wid > maxID then maxID = wid end
    cnt = cnt + 1
  end
  words[1] = DepTreeUtils.ROOT_MARK
  assert(cnt + 1 == maxID)
  
  return words
end

function DepTreeUtils.createVocab(inputFile, freqCut, ignoreCase, keepFreq)
  local fin = io.open(inputFile, 'r')
  local cnt = 0
  local wordVector = {}
  local wordFreq = {}
  while true do
    local line = fin:read()
    if line == nil then break end
    local fields = line:splitc('\t')
    assert(#fields == 2, 'MUST be two fields in dep file')
    local depWords = DepTreeUtils.parseDepStr(fields[2])
    local words = DepTreeUtils.dwords2words(depWords)
    for _, word in ipairs(words) do
      if ignoreCase then word = word:lower() end
      if wordFreq[word] then
        wordFreq[word] = wordFreq[word] + 1
      else
        wordFreq[word] = 1
        wordVector[#wordVector + 1] = word
      end
    end
    cnt = cnt + 1
  end
  xprintln('totally %d lines', cnt)
  fin:close()
  
  local wid = 1
  local word2idx = {}
  word2idx = {[DepTreeUtils.UNK_TOKEN] = wid}
  wid = wid + 1
  
  local uniqUNK = 0
  local freqs = { 0 }
  for _, wd in ipairs(wordVector) do
    if wordFreq[wd] > freqCut then
      word2idx[wd] = wid
      freqs[wid] = wordFreq[wd]
      wid = wid + 1
    else
      uniqUNK = uniqUNK + 1
      freqs[1] = freqs[1] + wordFreq[wd]
    end
  end
  
  local vocabSize = wid - 1
  
  local idx2word = {}
  for wd, i in pairs(word2idx) do
    idx2word[i] = wd
  end
  
  local vocab = {word2idx = word2idx, idx2word = idx2word, 
    freqCut = freqCut, ignoreCase = ignoreCase, 
    keepFreq = keepFreq, UNK = word2idx[DepTreeUtils.UNK_TOKEN],
    UNK_TOKEN = DepTreeUtils.UNK_TOKEN}
  if keepFreq then
    vocab['freqs'] = freqs
    vocab['uniqUNK'] = uniqUNK
    printf('freqs size %d\n', #freqs)
  end
  
  assert(vocabSize == table.len(word2idx))
  printf('original #words %d, after cut = %d, #words %d\n', #wordVector, freqCut, vocabSize)
  vocab['nvocab'] = vocabSize
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

function DepTreeUtils.getWordID(vocab, word)
  word = vocab.ignoreCase and word:lower() or word
  return vocab.word2idx[word] or vocab.UNK
end

function DepTreeUtils.deptree2hdf5(depFile, h5out, label, vocab, maxLen)
  local fin = io.open(depFile, 'r')
  local lineNo, offset, delCnt = 0, 1, 0
  local x_data = {}
  local y_data = {}
  local index = {}
  
  local gxdata = string.format('/%s/x_data', label)
  local gydata = string.format('/%s/y_data', label)
  local gindex = string.format('/%s/index', label)
  local isFirst = true
  
  local xOpt = hdf5.DataSetOptions()
  xOpt:setChunked(1024*50*10, 4)
  -- xOpt:setDeflate(1)
  local yOpt = hdf5.DataSetOptions()
  yOpt:setChunked(1024*50*10)
  -- yOpt:setDeflate(1)
  local iOpt = hdf5.DataSetOptions()
  iOpt:setChunked(1024*10, 2)
  -- iOpt:setDeflate(1)
  
  local function appendData()
    local x_data_ = torch.IntTensor(x_data)
    local y_data_ = torch.IntTensor(y_data)
    local index_ = torch.IntTensor(index)
    if not isFirst then
      h5out:append(gxdata, x_data_, xOpt)
      h5out:append(gydata, y_data_, yOpt)
      h5out:append(gindex, index_, iOpt)
    else
      h5out:write(gxdata, x_data_, xOpt)
      h5out:write(gydata, y_data_, yOpt)
      h5out:write(gindex, index_, iOpt)
      isFirst = false
    end
  end
  
  while true do
    local line = fin:read()
    if line == nil then break end
    local fields = line:splitc('\t')
    assert(#fields == 2, 'MUST be two fields in dep file')
    local depGraph = DepGraph()
    local depWords = DepTreeUtils.parseDepStr(fields[2])
    if #depWords <= maxLen then
      if depGraph:build(depWords) then
        depGraph:sortChildren()
        local lrepr = depGraph:getLinearRepr()
        for _, rep in ipairs(lrepr) do
          table.insert(x_data, 
            {DepTreeUtils.getWordID(vocab, rep[1]), rep[2], rep[3], rep[4]})
          table.insert(y_data, DepTreeUtils.getWordID(vocab, rep[5]))
        end
        local rlen = #lrepr
        table.insert(index, {offset, rlen})
        offset = offset + rlen
        lineNo = lineNo + 1
        if lineNo % 50000 == 0 then
          appendData()
          x_data = {}
          y_data = {}
          index = {}
          printf('[deptree2hdf5] line count = %d\n', lineNo)
        end
      else
        print('load graph failed!')
        print(fields[2])
      end
    else
      delCnt = delCnt + 1
    end
  end
  
  if #x_data > 0 then
    appendData()
  end
  
  printf('[%s] delete %d sentences\n', label, delCnt)
  printf('[%s] totally %d lines\n', label, lineNo)
  
  fin:close()
end

function DepTreeUtils.deptree2hdf5Bidirectional(depFile, h5out, label, vocab, maxLen)
  local lineNo, offset, delCnt = 0, 1, 0
  local loffset = 1
  local x_data = {}
  local y_data = {}
  local index = {}
  
  local gxdata = string.format('/%s/x_data', label)
  local gydata = string.format('/%s/y_data', label)
  local gindex = string.format('/%s/index', label)
  
  local l_data = {}
  local lindex = {}
  local gldata = string.format('/%s/l_data', label)
  local glindex = string.format('/%s/lindex', label)
  
  local isFirst = true
  
  local xOpt = hdf5.DataSetOptions()
  xOpt:setChunked(1024*50*10, 5)
  -- xOpt:setDeflate(1)
  local yOpt = hdf5.DataSetOptions()
  yOpt:setChunked(1024*50*10)
  -- yOpt:setDeflate(1)
  local iOpt = hdf5.DataSetOptions()
  iOpt:setChunked(1024*10, 2)
  -- iOpt:setDeflate(1)
  local lOpt = hdf5.DataSetOptions()
  lOpt:setChunked(1024*50*10, 2)
  local liOpt = hdf5.DataSetOptions(1024*10, 2)
  liOpt:setChunked(1024*10, 2)
  
  local function appendData()
    local x_data_ = torch.IntTensor(x_data)
    local y_data_ = torch.IntTensor(y_data)
    local index_ = torch.IntTensor(index)
    local l_data_ = torch.IntTensor(l_data)
    local lindex_ = torch.IntTensor(lindex)
    
    if not isFirst then
      h5out:append(gxdata, x_data_, xOpt)
      h5out:append(gydata, y_data_, yOpt)
      h5out:append(gindex, index_, iOpt)
      h5out:append(gldata, l_data_, lOpt)
      h5out:append(glindex, lindex_, liOpt)
    else
      h5out:write(gxdata, x_data_, xOpt)
      h5out:write(gydata, y_data_, yOpt)
      h5out:write(gindex, index_, iOpt)
      h5out:write(gldata, l_data_, lOpt)
      h5out:write(glindex, lindex_, liOpt)
      isFirst = false
    end
  end
  
  local function depIter(depFile)
    local fin = io.open(depFile)
    
    return function()
      local line = fin:read()
      if line ~= nil then
        local fields = line:splitc('\t')
        assert(#fields == 2, 'MUST be two fields in dep file')
        local depWords = DepTreeUtils.parseDepStr(fields[2])
        return depWords
      end
    end
  end
  
  local iter = depIter(depFile)
  for depWords in iter do
    local depGraph = DepGraph()
    if #depWords <= maxLen then
      if depGraph:build(depWords) then
        depGraph:sortChildren()
        local lrepr, lchildren = depGraph:getBidirectionalLinearRepr()
        for _, rep in ipairs(lrepr) do
          table.insert(x_data, 
            {DepTreeUtils.getWordID(vocab, rep[1]), rep[2], rep[3], rep[4], rep[5]})
          table.insert(y_data, DepTreeUtils.getWordID(vocab, rep[6]))
        end
        local rlen = #lrepr
        table.insert(index, {offset, rlen})
        offset = offset + rlen
        
        for _, lc in ipairs(lchildren) do
          table.insert(l_data, {DepTreeUtils.getWordID(vocab, lc[1]), lc[2]})
        end
        local llen = #lchildren
        table.insert(lindex, {loffset, llen})
        loffset = loffset + llen
        
        lineNo = lineNo + 1
        if lineNo % 50000 == 0 then
          appendData()
          x_data = {}
          y_data = {}
          index = {}
          l_data = {}
          lindex = {}
          printf('[deptree2hdf5] line count = %d\n', lineNo)
        end
      else
        print('load graph failed!')
        print(fields[2])
      end
    else
      delCnt = delCnt + 1
    end
    
  end
  
  if #x_data > 0 then
    appendData()
  end
  
  printf('[%s] delete %d sentences\n', label, delCnt)
  printf('[%s] totally %d lines\n', label, lineNo)
end

require 'conllxutils'

function DepTreeUtils.createVocabCoNLLX(inputFile, freqCut, ignoreCase, keepFreq, normalizeNumber)
  local iter = CoNLLXUtils.conllxIterator(inputFile, normalizeNumber)
  local cnt = 0
  local wordVector = {}
  local wordFreq = {}
  
  for depWords in iter do
    local words = DepTreeUtils.dwords2words(depWords)
    for _, word in ipairs(words) do
      if ignoreCase then word = word:lower() end
      if wordFreq[word] then
        wordFreq[word] = wordFreq[word] + 1
      else
        wordFreq[word] = 1
        wordVector[#wordVector + 1] = word
      end
    end
    cnt = cnt + 1
  end
  xprintln('totally %d trees', cnt)
  
  local wid = 1
  local word2idx = {}
  word2idx = {[DepTreeUtils.UNK_TOKEN] = wid}
  wid = wid + 1
  
  local uniqUNK = 0
  local freqs = { 0 }
  for _, wd in ipairs(wordVector) do
    if wordFreq[wd] > freqCut then
      word2idx[wd] = wid
      freqs[wid] = wordFreq[wd]
      wid = wid + 1
    else
      uniqUNK = uniqUNK + 1
      freqs[1] = freqs[1] + wordFreq[wd]
    end
  end
  
  local vocabSize = wid - 1
  
  local idx2word = {}
  for wd, i in pairs(word2idx) do
    idx2word[i] = wd
  end
  
  local vocab = {word2idx = word2idx, idx2word = idx2word, 
    freqCut = freqCut, ignoreCase = ignoreCase, 
    keepFreq = keepFreq, UNK = word2idx[DepTreeUtils.UNK_TOKEN],
    UNK_TOKEN = DepTreeUtils.UNK_TOKEN}
  if keepFreq then
    vocab['freqs'] = freqs
    vocab['uniqUNK'] = uniqUNK
    printf('freqs size %d\n', #freqs)
  end
  
  vocab.normalizeNumber = normalizeNumber
  
  assert(vocabSize == table.len(word2idx))
  printf('original #words %d, after cut = %d, #words %d\n', #wordVector, freqCut, vocabSize)
  vocab['nvocab'] = vocabSize
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

function DepTreeUtils.conllx2hdf5(depFile, h5out, label, vocab, maxLen)
  local lineNo, offset, delCnt = 0, 1, 0
  local x_data = {}
  local y_data = {}
  local index = {}
  
  local gxdata = string.format('/%s/x_data', label)
  local gydata = string.format('/%s/y_data', label)
  local gindex = string.format('/%s/index', label)
  local isFirst = true
  
  local xOpt = hdf5.DataSetOptions()
  xOpt:setChunked(1024*50*10, 4)
  -- xOpt:setDeflate(1)
  local yOpt = hdf5.DataSetOptions()
  yOpt:setChunked(1024*50*10)
  -- yOpt:setDeflate(1)
  local iOpt = hdf5.DataSetOptions()
  iOpt:setChunked(1024*10, 2)
  -- iOpt:setDeflate(1)
  
  local function appendData()
    local x_data_ = torch.IntTensor(x_data)
    local y_data_ = torch.IntTensor(y_data)
    local index_ = torch.IntTensor(index)
    if not isFirst then
      h5out:append(gxdata, x_data_, xOpt)
      h5out:append(gydata, y_data_, yOpt)
      h5out:append(gindex, index_, iOpt)
    else
      h5out:write(gxdata, x_data_, xOpt)
      h5out:write(gydata, y_data_, yOpt)
      h5out:write(gindex, index_, iOpt)
      isFirst = false
    end
  end
  
  local iter = CoNLLXUtils.conllxIterator(depFile, vocab.normalizeNumber)
  for depWords in iter do
    local depGraph = DepGraph()
    if #depWords <= maxLen then
      if depGraph:build(depWords) then
        depGraph:sortChildren()
        local lrepr = depGraph:getLinearRepr()
        for _, rep in ipairs(lrepr) do
          table.insert(x_data, 
            {DepTreeUtils.getWordID(vocab, rep[1]), rep[2], rep[3], rep[4]})
          table.insert(y_data, DepTreeUtils.getWordID(vocab, rep[5]))
        end
        local rlen = #lrepr
        table.insert(index, {offset, rlen})
        offset = offset + rlen
        lineNo = lineNo + 1
        if lineNo % 50000 == 0 then
          appendData()
          x_data = {}
          y_data = {}
          index = {}
          printf('[deptree2hdf5] line count = %d\n', lineNo)
        end
      else
        print('load graph failed!')
        print(fields[2])
      end
    else
      delCnt = delCnt + 1
    end
  end
  
  if #x_data > 0 then
    appendData()
  end
  
  printf('[%s] delete %d sentences\n', label, delCnt)
  printf('[%s] totally %d lines\n', label, lineNo)
end

function DepTreeUtils.conllx2hdf5Bidirectional(depFile, h5out, label, vocab, maxLen)
  local lineNo, offset, delCnt = 0, 1, 0
  local loffset = 1
  local x_data = {}
  local y_data = {}
  local index = {}
  
  local gxdata = string.format('/%s/x_data', label)
  local gydata = string.format('/%s/y_data', label)
  local gindex = string.format('/%s/index', label)
  
  local l_data = {}
  local lindex = {}
  local gldata = string.format('/%s/l_data', label)
  local glindex = string.format('/%s/lindex', label)
  
  local isFirst = true
  
  local xOpt = hdf5.DataSetOptions()
  xOpt:setChunked(1024*50*10, 5)
  -- xOpt:setDeflate(1)
  local yOpt = hdf5.DataSetOptions()
  yOpt:setChunked(1024*50*10)
  -- yOpt:setDeflate(1)
  local iOpt = hdf5.DataSetOptions()
  iOpt:setChunked(1024*10, 2)
  -- iOpt:setDeflate(1)
  local lOpt = hdf5.DataSetOptions()
  lOpt:setChunked(1024*50*10, 2)
  local liOpt = hdf5.DataSetOptions(1024*10, 2)
  liOpt:setChunked(1024*10, 2)
  
  local function appendData()
    local x_data_ = torch.IntTensor(x_data)
    local y_data_ = torch.IntTensor(y_data)
    local index_ = torch.IntTensor(index)
    local l_data_ = torch.IntTensor(l_data)
    local lindex_ = torch.IntTensor(lindex)
    
    if not isFirst then
      h5out:append(gxdata, x_data_, xOpt)
      h5out:append(gydata, y_data_, yOpt)
      h5out:append(gindex, index_, iOpt)
      h5out:append(gldata, l_data_, lOpt)
      h5out:append(glindex, lindex_, liOpt)
    else
      h5out:write(gxdata, x_data_, xOpt)
      h5out:write(gydata, y_data_, yOpt)
      h5out:write(gindex, index_, iOpt)
      h5out:write(gldata, l_data_, lOpt)
      h5out:write(glindex, lindex_, liOpt)
      isFirst = false
    end
  end
  
  local iter = CoNLLXUtils.conllxIterator(depFile, vocab.normalizeNumber)
  for depWords in iter do
    local depGraph = DepGraph()
    if #depWords <= maxLen then
      if depGraph:build(depWords) then
        depGraph:sortChildren()
        local lrepr, lchildren = depGraph:getBidirectionalLinearRepr()
        for _, rep in ipairs(lrepr) do
          table.insert(x_data, 
            {DepTreeUtils.getWordID(vocab, rep[1]), rep[2], rep[3], rep[4], rep[5]})
          table.insert(y_data, DepTreeUtils.getWordID(vocab, rep[6]))
        end
        local rlen = #lrepr
        table.insert(index, {offset, rlen})
        offset = offset + rlen
        
        for _, lc in ipairs(lchildren) do
          table.insert(l_data, {DepTreeUtils.getWordID(vocab, lc[1]), lc[2]})
        end
        local llen = #lchildren
        table.insert(lindex, {loffset, llen})
        loffset = loffset + llen
        
        lineNo = lineNo + 1
        if lineNo % 50000 == 0 then
          appendData()
          x_data = {}
          y_data = {}
          index = {}
          l_data = {}
          lindex = {}
          printf('[deptree2hdf5] line count = %d\n', lineNo)
        end
      else
        print('load graph failed!')
        print(fields[2])
      end
    else
      delCnt = delCnt + 1
    end
    
  end
  
  if #x_data > 0 then
    appendData()
  end
  
  printf('[%s] delete %d sentences\n', label, delCnt)
  printf('[%s] totally %d lines\n', label, lineNo)
end



