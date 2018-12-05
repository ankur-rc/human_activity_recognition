# HAR
Human Activity Recognition using smartphone data.  
Dataset: [UCI HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Benchmark results
```
python code/run_experiments.py -h
```
```
usage: Run models on the UCI HAR dataset. [-h] [--dataset DATASET]
                                          [--repeats REPEATS]
                                          [--models MODELS [MODELS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Root path to UCI HAR dataset
  --repeats REPEATS     No. of repeats for each model
  --models MODELS [MODELS ...]
                        List of models to evaluate on. Valid models are:
                        [lstm, cnn_lstm, conv_lstm, simple_cnn, wavenet_cnn]
```

### Output
Get **Precision**, **Recall** and **F1** score for the models across 'repeats' runs.  
Output will be generated in the folder '*results*' and models will be saved in '*models*'.

Also, tensorboard compatible training **logs** are generated for each run under the folder '*logs*' and subfolder 'model name'.

## Baseline results
```
python code/baseline.py
```

## Data Analysis
```
jupyter notebook code/notebooks/EDA.ipynb
```
