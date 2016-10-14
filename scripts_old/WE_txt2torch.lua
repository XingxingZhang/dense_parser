
include '../utils/shortcut.lua'

local function getShape(embedPath)
  local nrow , ncol = 0, 0
  local fin = io.open(embedPath, 'r')
  while true do
    local line = fin:read()
    if line == nil then break end
    if nrow == 0 then
      local fields = line:splitc(' \t')
      ncol = #fields
    end
    nrow = nrow + 1
  end
  fin:close()
  xprintln('word embedding file: nrow = %d, ncol = %d', nrow, ncol)
  
  return nrow, ncol
end

local function toTorch7(txtFile, torchFile)
  local nrow, ncol = getShape(txtFile)
  local embed = torch.FloatTensor(nrow, ncol-1)
  print('get embedding done!')
  local word2idx = {}
  local idx2word = {}
  local fin = io.open(txtFile, 'r')
  local cnt = 0
  while true do
    local line = fin:read()
    if line == nil then break end
    cnt = cnt + 1
    local fields = line:splitc(' \t')
    assert(#fields == ncol)
    table.insert(idx2word, fields[1])
    local vec = {}
    for i = 2, #fields do
      vec[#vec + 1]= tonumber(fields[i])
    end
    embed[cnt] = torch.FloatTensor(vec)
    if cnt % 10000 == 0 then print(cnt) end
  end
  xprintln('totaly lines %d\n', cnt)
  for idx, word in ipairs(idx2word) do
    word2idx[word] = idx
  end
  fin:close()
  xprintln('begin to save ...')
  torch.save(torchFile, {embed, word2idx, idx2word})
  xprintln('save done!')
end

local function main()
  toTorch7(arg[1], arg[2])
end

main()
