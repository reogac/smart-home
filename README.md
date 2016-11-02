##file list:

`modelling.py` : data preprocessing and model building utilities
`smart-aircon.py` : main program.
`read-sensors.py`, `merge-evaluation.py`: just for experiment.
`sensor-data.csv`: raw sensors data from one of the experiment offices.

##prediction evaluation:

5-fold cross validation was implemented and it can be done with following command:

`./smart-aircon evaluate` -t sensor-data.csv

type `./smart-aircon --help` for other commands
