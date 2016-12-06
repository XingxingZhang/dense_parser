# Dependency Parsing as Head Selection

This is an implmentation of the DeNSe (**De**pendency
**N**eural **Se**lection) parser described in [Dependency Parsing as Head Selection](https://arxiv.org/abs/1606.01280) 

# Dependencies
* [CUDA 7.0.28](http://www.nvidia.com/object/cuda_home_new.html)
* [Torch](https://github.com/torch)
* [torch-hdf5](https://github.com/deepmind/torch-hdf5)

You may also need to install some torch components.
```
luarocks install nn
luarocks install nngraph
luarocks install cutorch
luarocks install cunn
```
The parser was developed with an old version of torch (the version around Feb 2016).

# Run the Parser
The parser can parse text in conllx format (note that POS tags must be provided).
If the gold standard file is provided via `--gold`, the parse will also print out the UAS and LAS.
```
CUDA_VISIBLE_DEVICES=3 th dense_parser.lua --modelPath $model --classifierPath $classifier \
    --input $input --output $output --gold $input --mstalg Eisner 
```
Feel free to try scripts in `experiments/run_parser`.

# Get Train Dev Splits for German and Czech
Please refer to the `main` function of `conllx_scripts/split_dev.lua`

# Convert pre-trained embeddings
You need to convert glove vectors from text format to `t7` format.
```
conllx_scripts/extract_embed.lua -h
```

# Train an Unlabeled Parser
Without loss of generality, we use Czech as an example.

First, train the model with Adam algorithm using the script `experiments/czech/train.sh`
```
CUDA_VISIBLE_DEVICES=$ID th train.lua --useGPU \
    --model SelectNetPos \
    --seqLen 112 \
    --maxTrainLen 110 \
    --freqCut 1 \
    --nhid 300 \
    --nin 300 \
    --nlayers 2 \
    --dropout 0.35 \
    --recDropout 0.1 \
    --lr $lr \
    --train $train \
    --valid $valid \
    --test $test \
    --optimMethod Adam \
    --save $model \
    --batchSize 20 \
    --validBatchSize 20 \
    --maxEpoch 15 \
    --npin 40 \
    --evalType conllx \
    | tee $log
```
After we reach the convergence of Adam, we switch to plain SGD using `experiments/czech/tune.sh`, which can usually give us a slight improvement.
```
CUDA_VISIBLE_DEVICES=$ID th post_train.lua \
    --load $load \
    --save $model \
    --lr $lr \
    --maxEpoch 10 \
    --optimMethod SGD \
    | tee $log
```
Lastly, we use a MST algorithm to adjust the non-tree outputs with `experiments/czech/mst-post.sh`
```
CUDA_VISIBLE_DEVICES=3 th mst_postprocess.lua \
    --modelPath $model \
    --mstalg ChuLiuEdmonds \
    --validout $validout \
    --testout $testout | tee $log
```

# Train a labeled Parser
Based on the trained unlabled parser, we first generate training data for the labeled parser with `experiments/czech/gen_lbl_train.sh`
```
CUDA_VISIBLE_DEVICES=3 th train_labeled.lua --mode generate \
	--modelPath $model \
	--outTrainDataPath $outTrain \
	--inTrain $inTrain \
	--inValid $inValid \
	--inTest $inTest \
	--outValid $outValid \
	--outTest $outTest \
	--language Other | tee $log
```
Then we train the labeled parser actually an MLP with `experiments/czech/run_lbl.sh`
```
CUDA_VISIBLE_DEVICES=3 th train_labeled.lua --mode train \
	--useGPU \
	--snhids "1880,800,800,82" \
	--activ relu \
	--lr 0.01 \
	--optimMethod AdaGrad \
	--dropout 0.5 \
	--inDropout 0.05 \
	--batchSize 256 \
	--maxEpoch 20 \
	--ftype "|x|xe|xpe|" \
	--dataset $dataset \
	--inTrain $inTrain \
	--inValid $inValid \
	--inTest $inTest \
	--language Other \
	--save $model | tee $log
```

# Downloads
## Pre-trained Models
https://drive.google.com/file/d/0B6-YKFW-MnbORXdfMzRwVWt3UG8/view?usp=sharing
## Pre-trained Chinese Embeddings
https://drive.google.com/file/d/0B6-YKFW-MnbOMjdXSVlKTkFwR0E/view?usp=sharing


