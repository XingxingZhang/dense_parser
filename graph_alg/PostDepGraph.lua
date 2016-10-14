
local Vertex = torch.class('PDVertex')

function Vertex:__init(id, wd)
  self.id = id or -1
  self.wd = wd or 'UNKNOWN'
  self.adjList = {}
end

local DepGraph = torch.class('PostDepGraph')

function DepGraph:__init(sent)
  self.vertices = {[0] = PDVertex(0, '###root###')}
  for _, item in ipairs(sent) do
    local id, wd = item.p1, item.wd
    self.vertices[id] = PDVertex(id, wd)
  end
  self.size = #sent + 1
  
  for _, item in ipairs(sent) do
    local u = tonumber(item.p2)
    local v = tonumber(item.p1)
    -- printf('u = %d, v = %d, N = %d\n', u, v, self.size)
    table.insert( self.vertices[u].adjList, self.vertices[v] )
  end
end

function DepGraph:checkConnectivity()
  local visited = {}
  for i = 0, self.size-1 do visited[i] = false end
  local root = self.vertices[0]
  self:dfs(root, visited)
  for i = 0, self.size-1 do
    if visited[i] == false then return false end
  end
  return true
end

function DepGraph:dfs(u, visited)
  visited[u.id] = true
  for _, v in ipairs(u.adjList) do
    if not visited[v.id] then
      self:dfs(v, visited)
    end
  end
end

function DepGraph:isProjective()
  -- sort child
  for i = 0, self.size - 1 do
    local u = self.vertices[i]
    u.lchild = {}
    u.rchild = {}
    for _, v in ipairs(u.adjList) do
      if v.id < u.id then
        u.lchild[#u.lchild + 1] = v
      elseif v.id > u.id then
        u.rchild[#u.rchild + 1] = v
      else
        error('impossible!')
      end
    end
    table.sort(u.lchild, function(a, b) return a.id < b.id end)
    table.sort(u.rchild, function(a, b) return a.id < b.id end)
  end
  
  local root = self.vertices[0]
  self.dfs_id = 0
  self.is_proj = true
  self:proj_dfs(root)
  
  for i = 0, self.size - 1 do
    self.vertices[i].lchild = nil
    self.vertices[i].rchild = nil
  end
  
  return self.is_proj
end

function DepGraph:proj_dfs(u)
  if #u.lchild > 0 then
    for _, v in ipairs(u.lchild) do
      self:proj_dfs(v)
    end
  end
  
  -- printf('u.id = %d, visited id = %d\n', u.id, self.dfs_id)
  if u.id ~= self.dfs_id then self.is_proj = false end
  self.dfs_id = self.dfs_id + 1
  
  if #u.rchild > 0 then
    for _, v in ipairs(u.rchild) do
      self:proj_dfs(v)
    end
  end
end

function DepGraph:showSentence()
  local sents = {}
  for i = 0, self.size - 1 do
    sents[#sents + 1] = self.vertices[i].wd
  end
  print(table.concat(sents, ' '))
end

