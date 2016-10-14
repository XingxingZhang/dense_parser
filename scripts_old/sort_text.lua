
include '../utils/shortcut.lua'

local Sorter = {}

function Sorter.sortKBatches(lens, Kbatch)
  local idxs = {}
  local N = #lens
  for istart = 1, N, Kbatch do
    local iend = math.min(istart + Kbatch - 1, N)
    local subIdxs = {}
    for i = istart, iend do 
      table.insert(subIdxs, i) 
    end
    table.sort(subIdxs, function(a, b)
         return lens[b] < lens[a]  
      end)
    table.extend(idxs, subIdxs)
  end
  assert(#idxs == #lens)
  return idxs
end

function Sorter.sortBatches(lens, batchSize)
  local newIdxs = {}
  
  local len2idxs = {}
  local len2idxs_lens = {}
  for i, len in ipairs(lens) do
    local idxs = len2idxs[len]
    if idxs then
      table.insert(idxs, i)
    else
      len2idxs[len] = {i}
      table.insert(len2idxs_lens, len)
    end
  end
  
  local len2pos = {}
  for _, len in ipairs(len2idxs_lens) do
    len2pos[len] = 1
  end
  
  local pad = {}
  while true do
    local selectLen = -1
    local selectIdx = #lens + 1
    local istart, iend = -1, -1
    
    for _, len in ipairs(len2idxs_lens) do
      local pos = len2pos[len]
      local idxs = len2idxs[len]
      if pos <= #idxs and idxs[pos] < selectIdx then
        selectIdx = idxs[pos]
        selectLen = len
        istart, iend = pos, math.min(pos + batchSize - 1, #idxs)
      end
    end
    
    if selectLen == -1 then break end
    local sIdxs = len2idxs[selectLen]
    if iend - istart + 1 == batchSize then
      for i = istart, iend do
        newIdxs[#newIdxs + 1] = sIdxs[i]
      end
    else
      for i = istart, iend do
        pad[#pad + 1] = sIdxs[i]
      end
    end
    len2pos[selectLen] = iend + 1
  end -- end while
  
  table.sort(pad, function(a, b) 
      return lens[b] < lens[a]
    end)
  table.extend(newIdxs, pad)
  
  return newIdxs
end

local function sort_train()
  infile = '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.100.train'
  outfile = '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.100.sort.train'
  printf('sort %s\n', infile)
  local fin = io.open(infile, 'r')
  lines = {}
  lens = {}
  local cnt = 0
  while true do
    local line = fin:read()
    table.insert(lines, line)
    if line == nil then break end
    words = line:splitc(' \t\r\n')
    table.insert(lens, #words)
    cnt = cnt + 1
    if cnt % 10000 == 0 then print(cnt) end
  end
  print(cnt)
  print(#lines)
  for i, len in ipairs(lens) do
    print(len)
    if i == 10 then break end
  end
  local newIdxs = Sorter.sortBatches(lens, 64)
  local fout = io.open(outfile, 'w')
  for i, idx in ipairs(newIdxs) do
    fout:write(lines[idx] .. '\n')
  end
  fout:close()
  fin:close()
end

-- sort_train()
local function sort_valid()
  infile = '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.100.valid'
  outfile = '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.100.sort.valid'
  printf('sort %s\n', infile)
  local fin = io.open(infile, 'r')
  lines = {}
  lens = {}
  local cnt = 0
  while true do
    local line = fin:read()
    table.insert(lines, line)
    if line == nil then break end
    words = line:splitc(' \t\r\n')
    table.insert(lens, #words)
    cnt = cnt + 1
    if cnt % 10000 == 0 then print(cnt) end
  end
  print(cnt)
  print(#lines)
  for i, len in ipairs(lens) do
    print(len)
    if i == 10 then break end
  end
  local newIdxs = Sorter.sortKBatches(lens, 4000)
  local fout = io.open(outfile, 'w')
  for i, idx in ipairs(newIdxs) do
    fout:write(lines[idx] .. '\n')
  end
  fout:close()
  fin:close()
end

sort_valid()

