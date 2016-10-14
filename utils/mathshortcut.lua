
require 'torch'

function safe_cdiv(a, b)
  local cb = b:clone()
  cb[cb:eq(0)] = 1
  return torch.cdiv(a, cb)
end

