
include '../utils/shortcut.lua'

local CoNLLXEval = {}

function CoNLLXEval.conllxLineIterator(infile)
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

function CoNLLXEval.eval(sysFile, goldFile)
  -- local punctTags ={ "``", "''", ".", ",", ":" }
  local punctTags ={ "``", "''", ".", ",", ":", "PU" }
  local punctTagSet = {}
  for _, pt in ipairs(punctTags) do
    punctTagSet[pt] = true
  end
  -- print(punctTagSet)
  
  local sysIter = CoNLLXEval.conllxLineIterator(sysFile)
  local goldIter = CoNLLXEval.conllxLineIterator(goldFile)
  local sen_cnt = 0
  local total, noPunctTotal = 0, 0
  local nUA, noPunctNUA = 0, 0
  local nLA, noPunctNLA = 0, 0
  
  for sysLines in sysIter do
    local goldLines = goldIter()
    assert(#sysLines == #goldLines, 'the sys sentence and the gold sentence should contain the same number of words')
    for i = 1, #sysLines do
      local sfields = sysLines[i]:splitc('\t ')
      local gfields = goldLines[i]:splitc('\t ')
      local sAid, gAid = tonumber(sfields[7]), tonumber(gfields[7])
      local sDep, gDep = sfields[8], gfields[8]
      if sAid == gAid then
        nUA = nUA + 1
        if sDep == gDep then nLA = nLA + 1 end
      end
      
      total = total + 1
      
      local gtag = gfields[5]
      if not punctTagSet[gtag] then
        noPunctTotal = noPunctTotal + 1
        if sAid == gAid then
          noPunctNUA = noPunctNUA + 1
          if sDep == gDep then noPunctNLA = noPunctNLA + 1 end
        end
      end
    end
    
    sen_cnt =  sen_cnt + 1
  end
  
  xprintln('totally %d sentences', sen_cnt)
  local LAS, UAS = nLA / total * 100, nUA / total * 100
  local noPunctLAS, noPunctUAS = noPunctNLA / noPunctTotal * 100, noPunctNUA / noPunctTotal * 100
  
  xprintln('==no punct==')
  xprintln('LAS = %.2f, UAS = %.2f', noPunctLAS, noPunctUAS)
  xprintln('==with punct==')
  xprintln('LAS = %.2f, UAS = %.2f', LAS, UAS)
  
  return LAS, UAS, noPunctLAS, noPunctUAS
end

return CoNLLXEval
