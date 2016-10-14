
include '../utils/shortcut.lua'

local function conllxLineIterator(infile)
  local fin = io.open(infile)
  local bufs = {}
  
  return function()
    while true do
      local line = fin:read()
      if line == nil then
        fin:close()
        break
      end
      line = line:trim()
      if line:len() == 0 then
        local rlines = {}
        for i, buf in ipairs(bufs) do
          rlines[i] = buf
        end
        table.clear(bufs)
        
        return rlines
      else
        table.insert(bufs, line)
      end
    end
  end
  
end

local function shuffle(inFile, outFile)
  local inIter = conllxLineIterator(inFile)
  local depTrees = {}
  for lines in inIter do
    depTrees[#depTrees + 1] = lines
  end
  local newIdxs = torch.randperm(#depTrees)
  local fout = io.open(outFile, 'w')
  for i = 1, newIdxs:size(1) do
    local idx = newIdxs[i]
    local lines = depTrees[idx]
    for _, line in ipairs(lines) do
      fout:write(string.format('%s\n', line))
    end
    fout:write('\n')
  end
  fout:close()
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== shuffle training set (CoNLL-X format) ======')
  cmd:option('--inFile', '', 'system output')
  cmd:option('--outFile', '', 'gold standard')
  cmd:option('--seed', 123, 'random seed for shuffle')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  print(opts)
  torch.manualSeed(opts.seed)
  shuffle(opts.inFile, opts.outFile)
end

main()


