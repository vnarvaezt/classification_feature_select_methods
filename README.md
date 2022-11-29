# Classification US 2020 elections

## Run the program
From the terminal run:   
```
python main.py --start start --end end
```
- start / end: allows to choose which steps of the workflow to run.
The options are: EXTRACT and PREPROCESSING 

Example
```bash
python main.py --start MODELLING --end MODELLING
```

## Details about the project structure:
####  1. Config: 
    - Config.py can be use to choose the features to keep in the preprocessing, the models to
      test and the path to save/read the data
      Options for the models to test are: SVM, BaggingClassifier, BalancedBaggingClassifier,
      RandomForestClassifier, GradientBoosting, XGBClassifier

#### 2. Preprocessing: 
    - src/preprocessing/transform_data.py: Initial preprocessing on the target and explanatory variables. 
    - src/preprocessing/preprocessing_x: filters X based on correlation and variance. 
    It additionally handle NAN values, variables with too few dictinct values
 
#### 3. Modelling:
    - src/models/split_scale.py: splits data into train and test and then scales X_train and X_test
    - src/models/modelling.py: runs the baseline model and any additional model specify in the config
 
#### 4. Notebooks:
    - notebooks/exploratory_analysis.ipynb: Explores all the raw data
    - notebooks/modelling.ipynb: Contains tests with different feature selection methods and a section for model explainability 
