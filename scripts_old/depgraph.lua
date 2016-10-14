------------------------------------
---- Inside the dependency tree ----
------------------------------------

include '../utils/xqueue.lua'
require 'dgvertex'
require 'dgedge'

local DepGraph = torch.class('DepGraph')

DepGraph.actions = {JL = 1, JR = 2, JLF = 3, JRF = 4}

-- build dependency graph
-- return true mean build successfully
function DepGraph:build(depWords)
  -- local depWords = DepTreeUtils.parseDepStr(depStr)
  local maxID = -1
  for _, dword in ipairs(depWords) do
    maxID = math.max(maxID, math.max(dword.p1, dword.p2))
  end
  self.size = maxID + 1
  self.vertices = {}
  for i = 1, self.size do self.vertices[i] = DGVertex() end
  for _, dword in ipairs(depWords) do
    local name, u, v = dword.rel, dword.p1 + 1, dword.p2 + 1
    local e = DGEdge(u, v, name)
    self.vertices[u].v = u
    self.vertices[u].tok = dword.w1
    self.vertices[v].v = v
    self.vertices[v].tok = dword.w2
    table.insert(self.vertices[u].adjList, e)
    
    if u == 1 and dword.w1 == DepTreeUtils.ROOT_MARK then
      self.root = u
    end
  end
  
  if self.root ~= 1 then return false end
  for i = 1, self.size do
    if self.vertices[i]:isEmpty() then return false end
  end
  
  return true
end

-- sort children of one node according to their distance
function DepGraph:sortChildren()
  local Q = XQueue(self.size)
  Q:push({self.root, 0})
  while not Q:isEmpty() do
    local u, level = unpack( Q:pop() )
    local curV = self.vertices[u]
    -- print(curV.v, curV.tok, level)
    curV.leftChildren = {}
    curV.rightChildren = {}
    for _, e in ipairs( curV.adjList ) do
      local v = e.v
      local nxV = self.vertices[v]
      if nxV.v < u then
        table.insert(curV.leftChildren, nxV)
      else
        table.insert(curV.rightChildren, nxV)
      end
    end
    table.sort(curV.leftChildren, function(vet1, vet2)
        return vet1.v > vet2.v
      end)
    table.sort(curV.rightChildren, function(vet1, vet2)
        return vet1.v < vet2.v
      end)
    for _, ch in ipairs(curV.leftChildren) do
      Q:push({ch.v, level + 1})
    end
    for _, ch in ipairs(curV.rightChildren) do
      Q:push({ch.v, level + 1})
    end
  end
end

function DepGraph:getLinearRepr()
  local lrepr = {}
  local Q = XQueue(self.size)
  Q:push(self.vertices[self.root])
  local bfsOrder = 1
  while not Q:isEmpty() do
    local u = Q:pop()
    u.bfsID = bfsOrder
    bfsOrder = bfsOrder + 1
    if u.v ~= self.root then
      table.insert(lrepr, {u.dependencyVertex.tok,
          u.action, u.dependencyVertex.bfsID, u.bfsID, u.tok})
    end
    -- assign dependency path for left children
    for i, v in ipairs(u.leftChildren) do
      if i == 1 then
        v.dependencyVertex = u
        v.action = DepGraph.actions.JL
      else
        v.dependencyVertex = u.leftChildren[i - 1]
        v.action = DepGraph.actions.JLF
      end
      Q:push(v)
    end
    -- assign dependency path for right children
    for i, v in ipairs(u.rightChildren) do
      if i == 1 then
        v.dependencyVertex = u
        v.action = DepGraph.actions.JR
      else
        v.dependencyVertex = u.rightChildren[i - 1]
        v.action = DepGraph.actions.JRF
      end
      Q:push(v)
    end
  end
  
  return lrepr
end

-- get bi-directional linear represetation
function DepGraph:getBidirectionalLinearRepr()
  local lrepr = {}
  local lchildren = {}
  
  local Q = XQueue(self.size)
  Q:push(self.vertices[self.root])
  local bfsOrder = 1
  while not Q:isEmpty() do
    local u = Q:pop()
    u.bfsID = bfsOrder
    bfsOrder = bfsOrder + 1
    if u.v ~= self.root then
      table.insert(lrepr, {u.dependencyVertex.tok,
          u.action, u.dependencyVertex.bfsID, u.bfsID, u.leftCxtPos, u.tok})
    end
    -- assign dependency path for left children
    for i, v in ipairs(u.leftChildren) do
      if i == 1 then
        v.dependencyVertex = u
        v.action = DepGraph.actions.JL
      else
        v.dependencyVertex = u.leftChildren[i - 1]
        v.action = DepGraph.actions.JLF
      end
      Q:push(v)
    end
    -- assign dependency path for right children
    for i, v in ipairs(u.rightChildren) do
      if i == 1 then
        v.dependencyVertex = u
        v.action = DepGraph.actions.JR
        if #u.leftChildren ~= 0 then
          for j = #u.leftChildren, 1, -1 do
            local lc = u.leftChildren[j]
            lchildren[#lchildren + 1] = {lc.tok, j == #u.leftChildren and 1 or 0}
          end
          v.leftCxtPos = #lchildren
        end
      else
        v.dependencyVertex = u.rightChildren[i - 1]
        v.action = DepGraph.actions.JRF
      end
      Q:push(v)
    end
  end
  
  return lrepr, lchildren
end


