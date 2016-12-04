
include '../utils/shortcut.lua'

local function extractTxtWE(vocabF, WEPath, WEOutPath)
  local vocab = torch.load(vocabF)
  print(table.keys(vocab))
  
  local fin = io.open(WEPath)
  local msize, nsize = 0, 0
  local cnt = 0
  local wetable = {}
  local idx2word = {}
  while true do
    local line = fin:read()
    if line == nil then break end
    local fields = line:splitc(' \t')
    local width = #fields - 1
    if nsize == 0 then
      nsize = width
    else
      assert(nsize == width)
    end
    local word = fields[1]
    if vocab.word2idx[word] ~= nil then
      msize = msize + 1
      idx2word[msize] = word
      local v = {}
      for i = 2, width + 1 do
        table.insert(v, tonumber(fields[i]))
      end
      table.insert(wetable, v)
    end
    
    cnt = cnt + 1
    if cnt % 10000 == 0 then
      printf('cnt = %d\n', cnt)
    end
  end
  print('totally ' .. msize .. ' lines remain')
  
  local word2idx = {}
  for i, w in pairs(idx2word) do
    word2idx[w] = i
  end
  print(#word2idx, #idx2word)
  
  local final_we = torch.FloatTensor(wetable)
  print 'begin to save'
  torch.save(WEOutPath, {final_we, word2idx, idx2word})
  print( 'save done at ' .. WEOutPath)
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('--vocab', 'ptb.train.tmp.vocab.t7', 'path for vocab file')
  cmd:option('--wepath', 'glove.840B.300d.txt', 'glove vectors')
  cmd:option('--weoutpath', 'ptb.glove.840B.300d.t7', 'output embedding file')
  local opts = cmd:parse(arg)
  
  extractTxtWE(opts.vocab, opts.wepath, opts.weoutpath)
end

main()

