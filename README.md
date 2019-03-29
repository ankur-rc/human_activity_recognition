# HAR
Human Activity Recognition using smartphone data.  
Dataset: [UCI HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

| Algorithm | Precision            | Recall               | F1-Score             |
|-----------|----------------------|----------------------|----------------------|
| LSTM      | 0.89195 (+/-0.00917) | 0.88775 (+/-0.00913) | 0.88788 (+/-0.00881) |
| CNN-LSTM  | 0.89570 (+/-0.00866) | 0.89121 (+/-0.01087) | 0.89127 (+/-0.01103) |
| ConvLSTM  | 0.90512 (+/-0.00701) | 0.90037 (+/-0.00921) | 0.90071 (+/-0.00907) |

![alt text](https://github.com/nautilusPrime/human_activity_recognition/blob/master/code/imgs/lstm.png?raw=true "LSTM")
![alt text](https://github.com/nautilusPrime/human_activity_recognition/blob/master/code/imgs/cnn_lstm.png?raw=true "CNN-LSTM")
![alt text](https://github.com/nautilusPrime/human_activity_recognition/blob/master/code/imgs/conv_lstm.png?raw=true "ConvLSTM")


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

## LSTM-CRF Model
```
jupyter notebook code/notebooks/data_loader.ipynb
jupyter notebook code/notebooks/lstm_crf.ipynb
```

## Baseline results
```
python code/baseline.py
```

## Data Analysis
```
jupyter notebook code/notebooks/EDA.ipynb
```

## Android App
Project build folder code/HARApp
