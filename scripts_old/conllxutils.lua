
local CoNLLXUtils = torch.class('CoNLLXUtils')

include '../utils/shortcut.lua'

CoNLLXUtils.ROOT_MARK = '###root###'

function CoNLLXUtils.normalizeNumber(str)
  local function match(str, pat)
    local istart, iend = str:find(pat)
    return istart ~= nil and iend ~= nil and iend - istart + 1 == str:len()
  end
  
  if match(str, '%d+') then
    return '<num>'
  elseif match(str, '%d+%.%d+') then
    return '<num>'
  elseif match(str, '%d[%d,]+') then
    return '<num>'
  else
    return str
  end
end

function CoNLLXUtils.conllxLines2dwords(lines, normalize)
  local words = {}
  for _, line in ipairs(lines) do
    local fields = line:splitc(' \t')
    assert(#fields == 10, 'MUST have ten columns')
    if normalize then
      words[tonumber(fields[1])] = CoNLLXUtils.normalizeNumber(fields[2])
    else
      words[tonumber(fields[1])] = fields[2]
    end
  end
  
  local dwords = {}
  for _, line in ipairs(lines) do
    local fields = line:splitc(' \t')
    local p2 = tonumber(fields[1])
    local p1 = tonumber(fields[7])
    local w2 = words[p2]
    local w1 = p1 == 0 and CoNLLXUtils.ROOT_MARK or words[p1]
    table.insert(dwords, {rel = fields[8], w1 = w1, 
        p1 = p1, w2 = w2, p2 = p2})
  end
  
  -- print('dwords')
  -- print(dwords)
  
  return dwords
end

function CoNLLXUtils.conllxIterator(infile, normalize)
  if normalize then
    print('Note Normalize Number')
  end
  
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
        local dwords = CoNLLXUtils.conllxLines2dwords(bufs, normalize)
        table.clear(bufs)
        
        return dwords
      else
        table.insert(bufs, line)
      end
    end
    
  end
end

function CoNLLXUtils.conllxLineIterator(infile)
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
