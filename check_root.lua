
require '.'
require 'shortcut'
require 'DepPosDataIter'

local function checkRoot(infile)
  local diter = DepPosDataIter.conllx_iter(infile)
  local cnt = 0
  local root_cnts = {[0] = 0, 0, 0, 0}
  for sent in diter do
    cnt = cnt + 1
    local root_cnt = 0
    for _, ditem in ipairs(sent) do
      local p2 = tonumber(ditem.p2)
      if p2 == 0 then
        root_cnt = root_cnt + 1
      end
    end
    if root_cnt ~= 1 then
      -- print(root_cnt)
      root_cnts[ root_cnt ] = root_cnts[ root_cnt ] + 1
    end
  end
  print(root_cnts)
  print(root_cnts[2] / cnt)
end

local function main()
  print '==for English=='
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post/valid.ori.dep')
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post/valid.dep')
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post/test.ori.dep')
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post/test.dep')
  
  print '==for Chinese=='
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/valid.ori.dep')
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/valid.dep')
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/test.ori.dep')
  checkRoot('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/test.dep')
end

main()
