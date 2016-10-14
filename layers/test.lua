
require 'nn'
require 'LookupTable_ft'

local function main()
   torch.manualSeed(1)
   print 'test LookupTable_ft'

   local mask = torch.Tensor({1, 0.01, 0.01, 1, 1})
   print 'mask'
   print(mask)

   local lt = LookupTable_ft(5, 5)
   lt:setUpdateMask(mask)
   local lt_ori = nn.LookupTable(5, 5)
   lt_ori.weight:copy(lt.weight)
   local x = torch.LongTensor({2, 3, 3, 4})
   print('x = ')
   print(x)

   local grad = torch.rand(4, 5)
   print 'OK grad'
   print(grad)

   local y = lt:forward(x)
   print 'lt y'
   print(y)
   print 'lt ori y'
   print(lt_ori:forward(x))
   lt:backward(x, grad)
   lt_ori(x, grad)

   print 'lt grad'
   print(lt.gradWeight)
   print 'lt ori grad'
   print(lt_ori.gradWeight)
end

main()

