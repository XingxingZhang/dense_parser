
require 'torch'
require 'hdf5'
include '../utils/shortcut.lua'

local SortHDF5 = torch.class('SortHDF5')

function SortHDF5:__init(h5infile, h5outfile)
  self.h5in = hdf5.open(h5infile, 'r')
  self.h5out = hdf5.open(h5outfile, 'w')
end

-- sortCmd is an integer
-- 0 means no sorting
-- sortCmd > 0 means sort every k batches
-- sortCmd < 0 means sort training data by their length with respect to
-- original order of the first sentence
function SortHDF5:sortHDF5(sortCmd, batchSize)
  assert(sortCmd ~= 0, 'If there is no sorting, don\'t use this class!')
  assert(batchSize ~= nil, 'batchSize MUST be specified!')
  local lens = self:getLengths('train')
  -- print('get lengths of dataset done!')
  local idxs
  if sortCmd > 0 then
    local Kbatch = sortCmd * batchSize
    idxs = self:sortKBatches(lens, Kbatch)
    print '[sort algorithm] = sort k batches!'
  else
    idxs = self:sortBatches(lens, batchSize)
    print '[sort algorithm] = sort batches!'
  end
  self:writeSplit('train', idxs)
  print('sort training set done!')
  
  local function sortAll(splitLabel)
    local vlens = self:getLengths(splitLabel)
    local vidxs = self:sortKBatches(vlens, #vlens)
    self:writeSplit(splitLabel, vidxs)
  end
  
  -- for valid and test partition
  sortAll('valid')
  print('sort valid set done!')
  sortAll('test')
  print('sort test set done!')
  
  self.h5in:close()
  self.h5out:close()
end

function SortHDF5:getLengths(splitLabel)
  local index = self.h5in:read(string.format('/%s/index', splitLabel))
  local N = index:dataspaceSize()[1]
  local lens = {}
  for i = 1, N do
    local idx = index:partial({i, i}, {1, 2})
    local start, len = idx[1][1], idx[1][2]
    table.insert(lens, len)
  end
  assert(#lens == N, 'Number of length should be consistent!')

  return lens
end

function SortHDF5:writeSplit(splitLabel, idxs)
  local index = self.h5in:read(string.format('/%s/index', splitLabel))
  local x_data = self.h5in:read(string.format('/%s/x_data', splitLabel))
  local y_data = self.h5in:read(string.format('/%s/y_data', splitLabel))
  
  local offset, isFirst = 1, true
  local x_ts = {}
  local y_ts = {}
  local i_ts = {}
  
  local gxdata = string.format('/%s/x_data', splitLabel)
  local gydata = string.format('/%s/y_data', splitLabel)
  local gindex = string.format('/%s/index', splitLabel)
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
    local x_data_ = torch.IntTensor(x_ts)
    local y_data_ = torch.IntTensor(y_ts)
    local index_ = torch.IntTensor(i_ts)
    if not isFirst then
      self.h5out:append(gxdata, x_data_, xOpt)
      self.h5out:append(gydata, y_data_, yOpt)
      self.h5out:append(gindex, index_, iOpt)
    else
      self.h5out:write(gxdata, x_data_, xOpt)
      self.h5out:write(gydata, y_data_, yOpt)
      self.h5out:write(gindex, index_, iOpt)
      isFirst = false
    end
  end
  
  for sentCount, i in ipairs(idxs) do
    local idx = index:partial({i, i}, {1, 2})
    local start, len = idx[1][1], idx[1][2]
    local x = x_data:partial({start, start + len - 1}, {1, 4})
    local y = y_data:partial({start, start + len - 1})
    table.extend(x_ts, x:totable())
    table.extend(y_ts, y:totable())
    table.insert(i_ts, {offset, len})
    if sentCount % 50000 == 0 then
      appendData()
      x_ts = {}
      y_ts = {}
      i_ts = {}
      printf('write [%s] line count = %d\n', splitLabel, sentCount)
      collectgarbage()
    end
    
    offset = offset + len
  end
  
  if #x_ts > 0 then
    appendData()
  end
end

function SortHDF5:sortKBatches(lens, Kbatch)
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

function SortHDF5:sortBatches(lens, batchSize)
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
