
sysFile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5/tools/mstparser-2/experiment/dev-mst1storder.conll
goldFile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5/data/valid
th eval.lua --sysFile $sysFile --goldFile $goldFile
