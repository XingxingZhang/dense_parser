
local DGEdge = torch.class('DGEdge')
function DGEdge:__init(u, v, name)
  self.u = u
  self.v = v
  self.name = name
end
