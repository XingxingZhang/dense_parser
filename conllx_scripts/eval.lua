
local conllx_eval = require('conllx_eval')

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Evaluation Script for Dependency Parser ======')
  cmd:option('--sysFile', '', 'system output')
  cmd:option('--goldFile', '', 'gold standard')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  local conllx_eval = require('conllx_eval')
  conllx_eval.eval(opts.sysFile, opts.goldFile)
end

main()
