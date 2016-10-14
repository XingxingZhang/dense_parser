
include '../utils/shortcut.lua'

local CoNLLXEval = {}

function CoNLLXEval.toStd(sysFile, goldFile)
  local tmpFile = sysFile .. '__tmp__'
  local fin_s = io.open(sysFile)
  local fin_g = io.open(goldFile)
  local fout = io.open(tmpFile, 'w')
  
  while true do
    local sline = fin_s:read()
    local gline = fin_g:read()
    if sline == nil then assert(gline == nil) break end
    if sline:len() ~= 0 then
      local sfields = sline:splitc('\t')
      local gfields = gline:splitc('\t')
      assert(#sfields == #gfields and #sfields == 10)
      for i = 1, 6 do
        sfields[i] = gfields[i]
      end
      sline = table.concat(sfields, '\t')
    end
    fout:write(sline)
    fout:write('\n')
  end
  
  fin_s:close()
  fin_g:close()
  fout:close()
  
  return tmpFile
end

function CoNLLXEval.run(sysFile, goldFile, params)
  params = params or '-q'
  local eval_script_name = 'eval.pl'
  if paths.dirp('./conllx_scripts') then
    eval_script_name = './conllx_scripts/' .. eval_script_name
  end
  local cmd = string.format('perl %s -g %s -s %s %s', eval_script_name, goldFile, sysFile, params)
  local fin = io.popen(cmd, 'r')
	local s = fin:read('*a')
  -- print '----------------------'
	-- print(s)
  -- print '----------------------'
  local reg = 'Labeled   attachment score: %d+ / %d+ %* 100 = ([^%s]+) %%'
  local _, _, lval = s:find(reg)
  local LAS = tonumber(lval)
  reg = 'Unlabeled attachment score: %d+ / %d+ %* 100 = ([^%s]+) %%'
  local _, _, uval = s:find(reg)
  local UAS = tonumber(uval)
  
  return LAS, UAS
end

function CoNLLXEval.eval(sysFile, goldFile)
  local tmp_sysfile = CoNLLXEval.toStd(sysFile, goldFile)
  local LAS, UAS = CoNLLXEval.run(tmp_sysfile, goldFile)
  
  local LAS_punct, UAS_punct = CoNLLXEval.run(tmp_sysfile, goldFile, "-q -p")
  os.execute(string.format('rm %s', tmp_sysfile))
  
  xprintln('==no punct==')
  xprintln('LAS = %.2f, UAS = %.2f', LAS, UAS)
  xprintln('==with punct==')
  xprintln('LAS = %.2f, UAS = %.2f', LAS_punct, UAS_punct)
  
  return LAS_punct, UAS_punct, LAS, UAS
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Evaluation Script for Dependency Parser: conllx 2006 standard ======')
  cmd:option('--sysFile', '', 'system output')
  cmd:option('--goldFile', '', 'gold standard')
  
  return cmd:parse(arg)
end

local function main()
  -- local sysFile = '/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_gold_test_1.conll'
  -- local goldFile = '/disk/scratch/Dataset/conll/2006/zxx_version/czech/czech_gold_test.conll'
  local opts = getOpts()
  CoNLLXEval.eval(opts.sysFile, opts.goldFile)
end

if not package.loaded['conllx2006_eval'] then
	main()
else
	return CoNLLXEval
end
