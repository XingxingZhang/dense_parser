
include '../utils/shortcut.lua'

local ReplaceConllxField = {}

function ReplaceConllxField.replace(infile, outfile, index, value)
  local fin = io.open(infile)
  local fout = io.open(outfile, 'w')
  while true do
    local line = fin:read()
    if line == nil then break end
    line = line:trim()
    if line:len() ~= 0 then
      local fields = line:splitc('\t')
      assert(#fields == 10, 'MUST have 10 fields!')
      fields[index] = value
      fout:write( table.concat(fields, '\t') )
      fout:write('\n')
    else
      fout:write('\n')
    end
  end
  fin:close()
  fout:close()
end

local function main()
  ReplaceConllxField.replace('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/valid.dep', 
    'test.out',
    8,
    'ROOT')
end

if not package.loaded['replace_conllx_field'] then
  main()
end

return ReplaceConllxField
