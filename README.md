# ConfusionFlow

Necessary files: 
- 1 config file per model
- 1 log file per model

### Config
Each model's config file should be a `json` file containing the name of each class in the dataset along with the number of examples of each class in the validation or test set on which the model will be evaluated. View an example config file [here](https://github.com/haydenedelson/ConfusionFlowProject/blob/main/config.json). 

### Logs
To generate the necessary log files, import the `Logs` object from the `callback.py` file in the `confusionflow/` folder. Instantiate a `Logs` object, providing it the `x` and `y` values of the validation or test set on which you'd like to evaluate your model, as well as the appropriate loss function (e.g. `"categorical_crossentropy"` or `"sparse_categorical_crossentropy"`). Pass this `Logs` object as a callback to your model. After model training, use the object's `.export()` method, passing in a directory path and model name, to save the model's performance logs. If the directory path does not exist, it will be created.

### Running
To launch the ConfusionFlow dashboard, run the following command in your command line from the root of this repository:
```
python confusionflow/run.py [model_logs].json [model_config].json
```

To launch the dashboard with multiple models, include all the log files and config files in sequence, as shown below:
```
python confusionflow/run.py  [log_1].json [config_1].json [log_2].json [config_2].json [log_3].json [config_3].json
```

To test the dashboard with the provided demo data, run the following command:
```
python confusionflow/run.py demo/model_1.json demo/config.json demo/model_2.json demo/config.json
```
