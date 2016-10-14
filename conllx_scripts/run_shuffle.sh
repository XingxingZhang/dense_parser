
seed=1
infile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5-1/data/train.autopos
outfile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5-1/data/train.autopos.s$seed
th shuffle.lua --inFile $infile --outFile $outfile --seed $seed

seed=2
infile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5-2/data/train.autopos
outfile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5-2/data/train.autopos.s$seed
th shuffle.lua --inFile $infile --outFile $outfile --seed $seed

seed=3
infile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5-3/data/train.autopos
outfile=/disk/scratch/s1270921/depparse/iornn-depparse-1st-t5-3/data/train.autopos.s$seed
th shuffle.lua --inFile $infile --outFile $outfile --seed $seed

