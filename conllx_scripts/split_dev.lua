
include '../utils/shortcut.lua'

local function conllxLineIterator(infile)
  local fin = io.open(infile)
  local bufs = {}
  
  return function()
    while true do
      local line = fin:read()
      if line == nil then
        fin:close()
        break
      end
      line = line:trim()
      if line:len() == 0 then
        local rlines = {}
        for i, buf in ipairs(bufs) do
          rlines[i] = buf
        end
        table.clear(bufs)
        
        return rlines
      else
        table.insert(bufs, line)
      end
    end
  end
  
end

local function get_corpus_information(train_file, maxlen)
  local train_sents = {}
  local train_iter = conllxLineIterator(train_file)
  local word_cnt, sent_cnt = 0, 0
  local sent_len = { max = 0, avg = 0, min = 123456789, over_cnt = 0, maxlen = maxlen }
  for sent in train_iter do
    sent_cnt = sent_cnt + 1
    local nword = #sent
    word_cnt = word_cnt + nword
    sent_len.avg = sent_len.avg + nword
    sent_len.max = math.max(sent_len.max, nword)
    sent_len.min = math.min(sent_len.min, nword)
    if nword > maxlen then
      sent_len.over_cnt = sent_len.over_cnt + 1
    end
  end
  sent_len.avg = sent_len.avg / sent_cnt
  print '==training set information=='
  printf('word cnt = %d, sent cnt = %d\n', word_cnt, sent_cnt)
  print 'Sentence Length:'
  print(sent_len)
  print(sent_len.over_cnt / sent_cnt)
  sent_len.sent_cnt = sent_cnt
  
  return sent_len
end

local function split_dev(train_file, test_file, out_file)
  local maxlen = 110
  print '==Corpus Information for Train=='
  local train_info = get_corpus_information(train_file, maxlen)
  print(train_info)
  
  print '==Corpus Information for Test=='
  local test_info = get_corpus_information(test_file, maxlen)
  print(test_info)
  
  local out_train_file = out_file .. 'train.conll'
  local out_dev_file = out_file .. 'dev.conll'
  local out_test_file = out_file .. 'test.conll'
  assert(train_file ~= out_train_file, 'MUST have a different name!!!')
  assert(test_file ~= out_test_file, 'MUST have different name!!!')
  local dev_cnt = test_info.sent_cnt + 10
  local train_cnt = train_info.sent_cnt - dev_cnt
  
  local train_iter = conllxLineIterator(train_file)
  local cnt = 0
  local fout_train = io.open(out_train_file, 'w')
  local fout_dev = io.open(out_dev_file, 'w')
  for sent in train_iter do
    cnt = cnt + 1
    local fout = cnt <= train_cnt and fout_train or fout_dev
    local label = cnt <= train_cnt and 'Train' or 'Dev'
    if #sent <= maxlen then
      for _, line in ipairs(sent) do
        fout:write(line .. '\n')
      end
      fout:write('\n')
    else
      printf('[%s] longer than %d\n', label, maxlen)
    end
  end
  fout_train:close()
  fout_dev:close()
  
  os.execute(string.format('cp %s %s', test_file, out_test_file))
  
  print '==Final Dataset=='
  print '===Train=='
  get_corpus_information(out_train_file, maxlen)
  print '===Dev==='
  get_corpus_information(out_dev_file, maxlen)
  print '===Test==='
  get_corpus_information(out_test_file, maxlen)
end

local function main()
  --[[
  split_dev('/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_pdt_train.conll',
    '/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_pdt_test.conll',
    '/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_gold_')
  --]]
  
  split_dev('/disk/scratch/Dataset/conll/2006/zxx_version/german/german_tiger_train.conll',
    '/disk/scratch/Dataset/conll/2006/zxx_version/german/german_tiger_test.conll',
    '/disk/scratch/Dataset/conll/2006/zxx_version/german/german_gold_')
end

main()