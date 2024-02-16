# Classification task for US 2020 elections
This project aims at testing diffent feature selection methods:
- [Boruta](https://pypi.org/project/Boruta/)
- [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- [Kbest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

These feature selection methods are tested along with the following classifiers: 
- SVM
- BaggingClassifier
- BalancedBaggingClassifier
- RandomForestClassifier
- GradientBoosting
- XGBClassifier

For the US election dataset, the best results were obtained using boruta and SVM classifier.


## Run the program
# Create a virtual environment named 'myenv'
```bash
python3 -m venv myenv
source myenv/bin/activate
# install all dependencies
pip install -r requirements.txt
``` 

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
