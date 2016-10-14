
train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.test
dataset=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.t7
th anonymize_ner.lua --train $train --valid $valid --test $test --dataset $dataset --totext


train=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.train
valid=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.valid
test=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.test
map=/afs/inf.ed.ac.uk/group/project/img2txt/encdec/dataset/newsela/V0V4_V1V4_V2V4_V3V4.aner.map.t7
th recover_anonymous.lua --train $train --valid $valid --test $test --map $map

