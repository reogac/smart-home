# Files that compose this project

| File                   | Description                                                   |
| -----------------------|:-------------------------------------------------------------:|
| `modelling.py`         | data preprocessing and model building utilities               | 
| `smart-aircon.py`      | main program                                                  | 
| `read-sensors.py`      | just for experiment.                                          | 
| `merge-evaluation.py`  | just for experiment                                           | 
| `sensor-data.csv`      | raw sensors data from one of the experiment offices           | 

#  Evaluate using 5-fold cross validation:

```bash
./smart-aircon evaluate -t sensor-data.csv
```

# Other commands

```bash
./smart-aircon --help
```
