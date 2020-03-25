# Commands

## Preprocessing

```
./opennmt/preprocess.py -train_src data/debug/text/trainA_000.text -train_tgt data/debug/code/trainA_000.code -valid_src data/debug/text/trainA_001.text -valid_tgt data/debug/code/trainA_001.code -save_data out/debug -dynamic_dict
```
TODO: Make this handle directories of text and code

## Training

```
./opennmt/train.py -config opennmt/config/small.train.yml -data out/debug -save_model out/model
```

To run on GPU, add the following:
```
-world_size 1 -gpu_ranks 0
```
and request for 1 GPU on the cluster.

## Prediction
```
 ./opennmt/translate.py -config opennmt/config/small.translate.yml -model out/model_step_10000.pt -src data/debug/text/trainA_004.text -output out/pred.txt
```

To run on GPU, add the following:
```
-gpu 1
```
and request for 1 GPU on the cluster. 
