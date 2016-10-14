
require 'torch'
require 'hdf5'
include '../utils/shortcut.lua'

local SortHDF5Bid = torch.class('SortHDF5Bidirectional', 'SortHDF5')

function SortHDF5Bid:writeSplit(splitLabel, idxs)
  print 'write bidirectional model!'
  local index = self.h5in:read(string.format('/%s/index', splitLabel))
  local x_data = self.h5in:read(string.format('/%s/x_data', splitLabel))
  local y_data = self.h5in:read(string.format('/%s/y_data', splitLabel))
  local l_data = self.h5in:read(string.format('/%s/l_data', splitLabel))
  local lindex = self.h5in:read(string.format('/%s/lindex', splitLabel))
  
  local offset, isFirst = 1, true
  local x_ts = {}
  local y_ts = {}
  local i_ts = {}
  
  local loffset = 1
  local l_ts = {}
  local li_ts = {}
  
  local gxdata = string.format('/%s/x_data', splitLabel)
  local gydata = string.format('/%s/y_data', splitLabel)
  local gindex = string.format('/%s/index', splitLabel)
  
  local gldata = string.format('/%s/l_data', splitLabel)
  local glindex = string.format('/%s/lindex', splitLabel)
  
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
    local x_data_ = torch.IntTensor(x_ts)
    local y_data_ = torch.IntTensor(y_ts)
    local index_ = torch.IntTensor(i_ts)
    local l_data_ = torch.IntTensor(l_ts)
    local lindex_ = torch.IntTensor(li_ts)
    if not isFirst then
      self.h5out:append(gxdata, x_data_, xOpt)
      self.h5out:append(gydata, y_data_, yOpt)
      self.h5out:append(gindex, index_, iOpt)
      self.h5out:append(gldata, l_data_, lOpt)
      self.h5out:append(glindex, lindex_, liOpt)
    else
      self.h5out:write(gxdata, x_data_, xOpt)
      self.h5out:write(gydata, y_data_, yOpt)
      self.h5out:write(gindex, index_, iOpt)
      self.h5out:write(gldata, l_data_, lOpt)
      self.h5out:write(glindex, lindex_, liOpt)
      isFirst = false
    end
  end
  
  for sentCount, i in ipairs(idxs) do
    local idx = index:partial({i, i}, {1, 2})
    local start, len = idx[1][1], idx[1][2]
    local x = x_data:partial({start, start + len - 1}, {1, 5})
    local y = y_data:partial({start, start + len - 1})
    table.extend(x_ts, x:totable())
    table.extend(y_ts, y:totable())
    table.insert(i_ts, {offset, len})
    
    local lidx = lindex:partial({i, i}, {1, 2})
    local lstart, llen = lidx[1][1], lidx[1][2]
    if llen ~= 0 then
      local lc = l_data:partial({lstart, lstart + llen - 1}, {1, 2})
      table.extend(l_ts, lc:totable())
    end
    table.insert(li_ts, {loffset, llen})
    
    if sentCount % 50000 == 0 then
      appendData()
      x_ts = {}
      y_ts = {}
      i_ts = {}
      l_ts = {}
      li_ts = {}
      printf('write [%s] line count = %d\n', splitLabel, sentCount)
      collectgarbage()
    end
    
    offset = offset + len
    loffset = loffset + llen
  end
  
  if #x_ts > 0 then
    appendData()
  end
end

