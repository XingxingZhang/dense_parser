
local DGVertex = torch.class('DGVertex')
function DGVertex:__init()
  self.v = -1
  self.tok = nil
  self.adjList = {}
  self.leftChildren = {}
  self.rightChildren = {}
  self.dependencyVertex = {}
  self.bfsID = -1
  self.action = -1
  
  -- this is for bidirectional model
  self.leftCxtPos = 0
end

function DGVertex:isEmpty()
  return self.v == -1 and self.tok == nil
end
