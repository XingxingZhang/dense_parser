
require '.'
require 'shortcut'
require 'PostDepGraph'
require 'DepPosDataIter'

local function check_proj(infile)
  local diter = DepPosDataIter.conllx_iter(infile)
  local cnt, tree_cnt, proj_cnt = 0, 0, 0
  for sent in diter do
    cnt = cnt + 1
    local dgraph = PostDepGraph(sent)
    if dgraph:checkConnectivity() then
      tree_cnt = tree_cnt + 1
      if dgraph:isProjective() then
        proj_cnt = proj_cnt + 1
      end
    end
  end
  printf('tree rate = %d / %d = %f\n', tree_cnt, cnt, tree_cnt/cnt)
  printf('projective rate = %d / %d = %f\n', proj_cnt, cnt, proj_cnt/cnt)
  print ''
end

local function main()
  --[[
  check_proj('/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_gold_train.conll')
  check_proj('/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_gold_dev.conll')
  check_proj('/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_gold_test.conll')
  
  check_proj('/disk/scratch/Dataset/conll/2006/zxx_version/german/german_gold_train.conll')
  check_proj('/disk/scratch/Dataset/conll/2006/zxx_version/german/german_gold_dev.conll')
  check_proj('/disk/scratch/Dataset/conll/2006/zxx_version/german/german_gold_test.conll')
  --]]

  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/UD/dataset/spanish/UD_Spanish/es-ud-train.conllx')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/UD/dataset/spanish/UD_Spanish/es-ud-dev.conllx')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/UD/dataset/spanish/UD_Spanish/es-ud-test.conllx')
  
  --[[
  -- /disk/scratch/Dataset/conll/2006/new_release/conll2006_ara-ces/data/arabic/PADT/treebank/arabic_PADT_train.conll
  print 'Arabic'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ara-ces/data/arabic/PADT/treebank/arabic_PADT_train.conll')
  
  print 'Bulgarian'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/bulgarian/bultreebank/train/bulgarian_bultreebank_train.conll')
  
  print 'Danish'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/danish/ddt/train/danish_ddt_train.conll')
  
  print 'Dutch'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/dutch/alpino/train/dutch_alpino_train.conll')
  
  print 'German'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/german/tiger/train/german_tiger_train.conll')
  
  print 'Japanese'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/japanese/verbmobil/train/japanese_verbmobil_train.conll')
  
  print 'Portuguese'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/portuguese/bosque/treebank/portuguese_bosque_train.conll')
  
  print 'Slovene'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/slovene/sdt/treebank/slovene_sdt_train.conll')
  
  print 'Spanish'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/spanish/cast3lb/train/spanish_cast3lb_train.conll')
  
  print 'Swedish'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/swedish/talbanken05/train/swedish_talbanken05_train.conll')
  
  print 'Turksh'
  check_proj('/disk/scratch/Dataset/conll/2006/new_release/conll2006_ten_lang/data/turkish/metu_sabanci/train/turkish_metu_sabanci_train.conll')
  --]]
  
  -- check_proj('/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/train.autopos')
  -- check_proj('/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/valid.autopos')
  -- check_proj('/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/test.autopos')
  --[[
  print ''
  print '== this is for English =='
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post_larger/valid.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post_larger/valid.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post_larger/test.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_post_larger/test.dep')
  
  print '== this is for Eisner algorithm =='
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger/valid.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger/valid.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger/test.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger/test.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger2/valid.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger2/valid.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger2/test.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/mst_eisner_larger2/test.dep')
  
  print ''
  print '== this is for Chinese =='
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/valid.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/valid.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/test.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_post/test.dep')
  
  print '== this is for Eisner algorithm =='
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner/valid.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner/valid.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner/test.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner/test.dep')
  
  -- /disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner2/valid.dep
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner2/valid.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner2/valid.dep')
  
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner2/test.ori.dep')
  check_proj('/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/mst_eisner2/test.dep')
  --]]
end

main()
