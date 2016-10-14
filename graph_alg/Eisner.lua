
local EisnerAlg = torch.class('Eisner')

-- index starts from 1
function EisnerAlg:load(size, inEdges)
  self.size = size
  self.adjMat = {}
  for i = 1, size do self.adjMat[i] = {} end
  for _, e in ipairs(inEdges) do
    self.adjMat[e[1]][e[2]] = e[3]
  end
end

function EisnerAlg:solve()
  -- init states
  local E = {}
  local A = {}
  for i = 1, self.size do
    E[i] = {}
    A[i] = {}
    for j = 1, self.size do
      E[i][j] = { 0, 0, 0, 0 }
      A[i][j] = { -1, -1, -1, -1 }
    end
  end
  -- do dynamic programming
  for L = 1, self.size - 1 do
    for s = 1, self.size - L do
      local t = s + L
      -- E[s][t][3]
      for q = s, t - 1 do
        local w_ts = self.adjMat[t][s] or 0
        -- E[s][t][3] = math.max( E[s][t][3], E[s][q][2] + E[q+1][t][1] + w_ts )
        if E[s][t][3] == 0 or E[s][q][2] + E[q+1][t][1] + w_ts > E[s][t][3] then
          E[s][t][3] = E[s][q][2] + E[q+1][t][1] + w_ts
          A[s][t][3] = q
        end
      end
      -- E[s][t][4]
      for q = s, t - 1 do
        local w_st = self.adjMat[s][t] or 0
        -- E[s][t][4] = math.max( E[s][t][4], E[s][q][2] + E[q+1][t][1] + w_st )
        if E[s][t][4] == 0 or E[s][q][2] + E[q+1][t][1] + w_st > E[s][t][4] then
          E[s][t][4] = E[s][q][2] + E[q+1][t][1] + w_st
          A[s][t][4] = q
        end
      end
      -- E[s][t][1]
      for q = s, t - 1 do
        -- E[s][t][1] = math.max( E[s][t][1], E[s][q][1] + E[q][t][3] )
        if E[s][t][1] == 0 or E[s][q][1] + E[q][t][3] > E[s][t][1] then
          E[s][t][1] = E[s][q][1] + E[q][t][3]
          A[s][t][1] = q
        end
      end
      -- E[s][t][2]
      for q = s + 1, t do
        -- E[s][t][2] = math.max( E[s][t][2], E[s][q][4] + E[q][t][2] )
        if E[s][t][2] == 0 or E[s][q][4] + E[q][t][2] > E[s][t][2] then
          E[s][t][2] = E[s][q][4] + E[q][t][2]
          A[s][t][2] = q
        end
      end
    end
  end
  
  -- find the edges
  local cost = E[1][self.size][2]
  
  local edges = {}
  local function getEdges(s, t, tt)
    if s < t then
      local q = A[s][t][tt]
      if tt == 3 then
        if self.adjMat[t][s] then
          edges[#edges + 1] = {u = t, v = s, weight = self.adjMat[t][s]}
        end
        getEdges(s, q, 2)
        getEdges(q + 1, t, 1)
      elseif tt == 4 then
        if self.adjMat[s][t] then
          edges[#edges + 1] = {u = s, v = t, weight = self.adjMat[s][t]}
        end
        getEdges(s, q, 2)
        getEdges(q + 1, t, 1)
      elseif tt == 1 then
        getEdges(s, q, 1)
        getEdges(q, t, 3)
      elseif tt == 2 then
        getEdges(s, q, 4)
        getEdges(q, t, 2)
      end
    end
  end
  
  getEdges(1, self.size, 2)
  
  return cost, edges
end

-- this is for test
local function main()
  require '../utils/shortcut'
  local eisner = Eisner()
  local N, M
  local edges = {}
  N = io.read('*number')
  M = io.read('*number')
  --[[
4 9
1 2 9
3 2 30
4 2 11
1 3 10
2 3 20
4 3 0
1 4 9
2 4 3
3 4 30
  --]]
  for i = 1, M do
    local u, v, w
    u = io.read('*number')
    v = io.read('*number')
    w = io.read('*number')
    table.insert(edges, {u, v, w})
  end
  eisner:load(N, edges)
  print( eisner:solve() )
end

if not package.loaded['Eisner'] then
  main()
end

