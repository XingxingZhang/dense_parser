
local DataIter = torch.class('DepDataIter')

function DataIter.conllx_iter(infile)
  local fin = io.open(infile)
  
  return function()
    local items = {}
    while true do
      local line = fin:read()
      if line == nil then break end
      line = line:trim()
      if line:len() == 0 then
        break
      end
      local fields = line:splitc('\t')
      assert(#fields == 10, 'MUST have 10 fields')
      local item = {p1 = tonumber(fields[1]), wd = fields[2], 
        pos = fields[5], p2 = fields[7], rel = fields[8]}
      table.insert(items, item)
    end
    if #items > 0 then
      return items
    else
      fin:close()
    end
  end
end

function DataIter.getDataSize(infiles)
  local sizes = {}
  for _, infile in ipairs(infiles) do
    local size = 0
    local diter = DataIter.conllx_iter(infile)
    for ds in diter do
      size = size + 1
    end
    sizes[#sizes + 1] = size
  end
  
  return sizes
end

function DataIter.showVocab(vocab)
  for k, v in pairs(vocab) do
    xprint(k)
    if type(v) == 'table' then
      print ' -- table' 
    else
      print( ' -- ' .. tostring(v) )
    end
  end
end

function DataIter.createVocab(infile, ignoreCase, freqCut, maxNVocab)
  local wordFreq = {}
  local wordVec = {}
  local diter = DataIter.conllx_iter(infile)
  
  local function addwd(wd)
    local wd = ignoreCase and wd:lower() or wd
    local freq = wordFreq[wd]
    if freq == nil then
      wordFreq[wd] = 1
      wordVec[#wordVec + 1] = wd
    else
      wordFreq[wd] = freq + 1
    end
  end
  
  for sent in diter do
    addwd('###root###')
    for _, ditem in ipairs(sent) do
      addwd(ditem.wd)
    end
  end
  
  local idx2word
  if freqCut and freqCut >= 0 then
    idx2word = {}
    idx2word[#idx2word + 1] = 'UNK'
    for _, wd in ipairs(wordVec) do
      if wordFreq[wd] > freqCut then idx2word[#idx2word + 1] = wd end
    end
    
    printf('original word count = %d, after freq cut = %d, word count = %d\n', #wordVec, freqCut, #idx2word)
  end
  
  if maxNVocab and maxNVocab > 0 then
    if #idx2word > 0 then
      print( 'WARING: rewrote idx2word with maxNVocab = ' .. maxNVocab )
    end
    idx2word = {}
    idx2word[#idx2word + 1] = 'UNK'
    local wfs = {}
    for _, k in ipairs(wordVec) do table.insert(wfs, {k, wordFreq[k]}) end
    table.sort(wfs, function(x, y) return x[2] > y[2] end)
    local lfreq = -1
    for cnt,  kv in ipairs(wfs) do
      idx2word[#idx2word + 1] = kv[1]
      lfreq = kv[2]
      if cnt >= maxNVocab-1 then break end
    end
    printf('original word count = %d, after maxNVocab = %d, word count = %d, lowest freq = %d\n', #wordVec, maxNVocab, #idx2word, lfreq)
  end
  
  local word2idx = {}
  for i, w in ipairs(idx2word) do word2idx[w] = i end
  local vocab = {word2idx = word2idx, idx2word = idx2word,
    freqCut = freqCut, ignoreCase = ignoreCase, maxNVocab = maxNVocab,
    UNK_STR = 'UNK', UNK = word2idx['UNK'],
    ROOT_STR = '###root###', ROOT = word2idx['###root###']}
  vocab['nvocab'] = table.len(word2idx)
  
  DataIter.showVocab(vocab)
  
  return vocab
end

function DataIter.toBatch(sents, vocab, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = -1
  for _, sent in ipairs(sents) do if sent:size(1) > maxn then maxn = sent:size(1) end end
  assert(maxn ~= -1)
  local x = (torch.ones(maxn + 1, batchSize) * vocab.UNK):type(dtype)
  local x_mask = torch.zeros(maxn + 1, batchSize)
  local y = torch.zeros(maxn, batchSize):type(dtype)
  x[{ 1, {} }] = vocab.ROOT
  x_mask[{ 1, {} }] = 1
  for i, sent in ipairs(sents) do
    local slen = sent:size(1)
    x[{ {2, slen + 1}, i }] = sent[{ {}, 1 }]
    x_mask[{ {2, slen + 1}, i }] = 1
    y[{ {1, slen}, i }] = sent[{ {}, 2 }]
  end
  
  return x, x_mask, y
end

function DataIter.sent2dep(vocab, sent)
  local d = {}
  local word2idx = vocab.word2idx
  for _, ditem in ipairs(sent) do
    local wd = vocab.ignoreCase and ditem.wd:lower() or ditem.wd
    local wid = word2idx[wd] or vocab.UNK
    d[#d + 1] = {wid, ditem.p2 + 1}
  end
  return torch.Tensor(d), #d
end


function DataIter.createBatch(vocab, infile, batchSize, maxlen)
  maxlen = maxlen or 100
  local diter = DataIter.conllx_iter(infile)
  local isEnd = false
  
  return function()
    if not isEnd then
      
      local sents = {}
      for i = 1, batchSize do
        local sent = diter()
        if sent == nil then isEnd = true break end
        local s, len = DataIter.sent2dep(vocab, sent)
        if len <= maxlen then 
          sents[#sents + 1] = s
        else
          print ( 'delete sentence with length ' .. tostring(len) )
        end
      end
      if #sents > 0 then
        return DataIter.toBatch(sents, vocab, batchSize)
      end
      
    end
  end
end

function DataIter.createBatchSort(vocab, infile, batchSize, maxlen)
  maxlen = maxlen or 100
  local diter = DataIter.conllx_iter(infile)
  local all_sents = {}
  for sent in diter do
    local s, len = DataIter.sent2dep(vocab, sent)
    all_sents[#all_sents + 1] = s
  end
  -- print(all_sents[1])
  table.sort(all_sents, function(a, b)  return a:size(1) < b:size(1) end)
  
  local cnt = 0
  local ndata = #all_sents
  
  return function()
    
    local sents = {}
    for i = 1, batchSize do
      cnt = cnt + 1
      if cnt <= ndata then
        sents[#sents + 1] = all_sents[cnt]
      end
    end
    
    if #sents > 0 then
      return DataIter.toBatch(sents, vocab, batchSize)
    end
    
  end
end

function DataIter.loadAllSents(vocab, infile, maxlen)
  local diter = DataIter.conllx_iter(infile)
  local all_sents = {}
  local del_cnt = 0
  for sent in diter do
    local s, len = DataIter.sent2dep(vocab, sent)
    if len <= maxlen then
      all_sents[#all_sents + 1] = s
    else
      del_cnt = del_cnt + 1
    end
  end
  if del_cnt > 0 then
    printf( 'WARNING!!! delete %d sentences that longer than %d\n', del_cnt, maxlen)
  end
  
  return all_sents
end

function DataIter.createBatchShuffleSort(all_sents_, vocab, batchSize, sort_flag, shuffle)
  assert(sort_flag ~= nil and (shuffle == true or shuffle == false))
  
  local function shuffle_dataset(all_sents)
    local tmp_sents = {}
    local idxs = torch.randperm(#all_sents)
    for i = 1, idxs:size(1) do
      tmp_sents[#tmp_sents + 1] = all_sents[ idxs[i] ]
    end
    return tmp_sents
  end
  
  local all_sents
  if shuffle then
    all_sents = shuffle_dataset(all_sents_)
  end
  
  local len_idxs = {}
  for i, sent in ipairs(all_sents) do
    len_idxs[#len_idxs + 1] = {sent:size(1), i}
  end
  
  local kbatches = sort_flag * batchSize
  local new_idxs = {}
  local N = #len_idxs
  for istart = 1, N, kbatches do
    iend = math.min(istart + kbatches - 1, N)
    local tmpa = {}
    for i = istart, iend do
      tmpa[#tmpa + 1] = len_idxs[i]
    end
    table.sort(tmpa, function( a, b ) return a[1] < b[1] end)
    for _, tmp in ipairs(tmpa) do
      new_idxs[#new_idxs + 1] = tmp[2]
    end
  end
  
  local final_all_sents = {}
  for _, idx in ipairs(new_idxs) do
    final_all_sents[#final_all_sents + 1] = all_sents[idx]
  end
  
  local cnt, ndata = 0, #final_all_sents
  return function()
    
    local sents = {}
    for i = 1, batchSize do
      cnt = cnt + 1
      if cnt > ndata then break end
      sents[#sents + 1] = final_all_sents[cnt]
    end
    
    if #sents > 0 then
      return DataIter.toBatch(sents, vocab, batchSize)
    end
    
  end
end

local function main()
  --[[
  require '../utils/shortcut'
  local infile = '/Users/xing/Desktop/depparse/train.autopos'
  local diter = DepDataIter.conllx_iter(infile)
  local cnt = 0
  for item in diter do
    -- print(item)
    cnt = cnt + 1
    -- if cnt == 1 then break end
  end
  printf('totally %d sentences\n', cnt)
  --]]
  
  --[[
  require '../utils/shortcut'
  local infile = '/Users/xing/Desktop/depparse/train.autopos'
  local vocab = DepDataIter.createVocab(infile, true)
  --]]
  
  require '../utils/shortcut'
  local infile = '/Users/xing/Desktop/depparse/train.autopos'
  local vocab = DepDataIter.createVocab(infile, true, 1)
  print 'get vocab done!'
  
  local validfile = '/Users/xing/Desktop/depparse/train.autopos'
  local batchIter = DepDataIter.createBatch(vocab, validfile, 30, 100)
  local cnt = 0
  for x, x_maks, y in batchIter do
    cnt = cnt + 1
  end
  print( 'totally ' .. cnt )
  
  -- batchIter()
  -- batchIter()
end

if not package.loaded['DepDataIter'] then
  main()
end

