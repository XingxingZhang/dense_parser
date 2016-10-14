
local function normalize(infile, outfile)
  local embed, word2idx, idx2word = unpack( torch.load(infile) )
  local N = embed:size(1)
  for i = 1, N do
    local norm = embed[i]:norm()
    embed[i]:div(norm)
  end
  
  torch.save(outfile, {embed, word2idx, idx2word})
end

normalize(arg[1], arg[2])
