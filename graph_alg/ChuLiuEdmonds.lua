
local CLE = torch.class('ChuLiuEdmonds')

function CLE:load(size, inEdges)
  self.size = size
  self.edges = {}
  for i, e in ipairs(inEdges) do
    local edge = {u = e[1], v = e[2], weight = e[3]}
    edge.ori_u = edge.u
    edge.ori_v = edge.v
    edge.ori_weight = edge.weight
    table.insert(self.edges, edge)
  end
end

function CLE:solve(root, N)
  local cost = 0
  local INF = 1e308
  local inWeights = {}
  local inEdges = {}
  for i = 1, N do inWeights[i] = -INF end
  -- for i = 1, N do inWeights[i] = INF end
  -- find largest input edges
  for _, edge in ipairs(self.edges) do
    local u, v = edge.u, edge.v
    if inWeights[v] < edge.weight and u ~= v then
    -- if inWeights[v] > edge.weight and u ~= v then
      inWeights[v] = edge.weight
      inEdges[v] = edge
    end
  end
  -- check if there is no incomming edge
  for i = 1, N do
    if i ~= root and inWeights[i] == -INF then
    -- if i ~= root and inWeights[i] == INF then
      return -1
    end
  end
  inWeights[root] = 0
  -- find cycles
  local newid = {}
  local vis = {}
  for i = 1, N do
    newid[i] = -1
    vis[i] = -1
  end
  local nx_id = 1
  -- handle cycles
  for i = 1, N do
    cost = cost + inWeights[i]
    local v = i
    while vis[v] ~= i and newid[v] == -1 and v ~= root do
      vis[v] = i
      v = inEdges[v].u
    end
    if v ~= root and newid[v] == -1 then
      local u = inEdges[v].u
      while u ~= v do
        newid[u] = nx_id
        u = inEdges[u].u
      end
      newid[v] = nx_id
      nx_id = nx_id + 1
    end
  end
  if nx_id == 1 then  -- no cycle, done!
    inEdges[root] = nil
    local finalEdges = {}
    for _, edge in pairs(inEdges) do
      table.insert(finalEdges, edge)
    end
    return cost, finalEdges
  end
  local max_cycle_id = nx_id - 1
  -- assign other nodes
  for i = 1, N do
    if newid[i] == -1 then
      newid[i] = nx_id
      nx_id = nx_id + 1
    end
  end
  -- rebuild graph and backup old edge information
  local cp_edges = {}
  for i, edge in ipairs(self.edges) do
    local u, v = edge.u, edge.v
    cp_edges[i] = {u = u, v = v, weight = edge.weight}
    edge.u, edge.v = newid[u], newid[v]
    if newid[u] ~= newid[v] then edge.weight = edge.weight - inWeights[v] end
  end
  
  local sub_cost, sub_inEdges = self:solve( newid[root], nx_id - 1 )
  
  for i, edge in ipairs(self.edges) do
    edge.u = cp_edges[i].u
    edge.v = cp_edges[i].v
    edge.weight = cp_edges[i].weight
  end
  
  local finalEdges = {}
  for i, edge in ipairs(sub_inEdges) do
    table.insert(finalEdges, edge)
    local v = edge.v
    -- add edges in a circle
    if newid[v] <= max_cycle_id then
      local u = inEdges[v].u
      while u ~= v do
        table.insert(finalEdges, inEdges[u])
        u = inEdges[u].u
      end
    end
  end
  
  return cost + sub_cost, finalEdges
end

local function main()
  require '../utils/shortcut'
  --[[
4 6
0 6
4 6
0 0
7 20
1 2
1 3
2 3
3 4
3 1
3 2
  --]]
  --[[
4 6
0 6
4 6
0 0
7 20
1 2
1 3
2 3
3 4
3 1
3 2
4 3
0 0
1 0
0 1
1 2
1 3
4 1
2 3
  --]]
  -- read data
  
  --[[
  local function dist(p1, p2)
    return math.sqrt( (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]) )
  end
  
  local N, M
  local edges = {}
  local points = {}
  N = io.read('*number')
  M = io.read('*number')
  for i = 1, N do
    local x, y
    x = io.read('*number')
    y = io.read('*number')
    points[#points + 1] = {x, y}
  end
  -- print(points)
  for i = 1, M do
    local u, v
    u = io.read('*number')
    v = io.read('*number')
    edges[#edges + 1] = {u, v, dist(points[u], points[v])}
  end
  
  print(edges)
  --]]
  
--[[
6 9
1 2 5
1 3 6
2 3 1
3 5 2
4 2 5
4 5 3
5 2 3
5 6 2
6 4 3
--]]
  local N, M
  local edges = {}
  N = io.read('*number')
  M = io.read('*number')
  for i = 1, M do
    local u, v, w
    u = io.read('*number')
    v = io.read('*number')
    w = io.read('*number')
    table.insert(edges, {u, v, w})
  end
  
  local cle = ChuLiuEdmonds()
  cle:load(N, edges)
  local cost, out_edges = cle:solve(1, N)
  print 'get MST done!'
  print(cost)
  print(out_edges)
  table.sort(out_edges, function(a, b) return a.v < b.v end)
  print(out_edges)
end

if not package.loaded['ChuLiuEdmonds'] then
  main()
end


