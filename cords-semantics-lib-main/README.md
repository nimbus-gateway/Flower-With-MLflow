# Cords-Semantics

Cords-Semantics is a Python library designed for tagging the artifacts generated during MLflow runs and creating their semantic descriptions. It utlizes CORDS Ontology[https://cords.ie/assets/cords_ml_ontology.html] for Machine Learning. This library helps interpret the meta data of machine learning artifiacts in more meanigful way, and prepares the assets to be shared using IDSA protocol. 


## Features

- **Tagging Artifacts**: Provide the CORDS vocab in as a set of MLFlow tags to be used during the training.
- **MLFlow MetaData Extractions**: Provide interfaces to extract MLflow tags and other metadata to a JSON structure 
- **Generate Semantic Description**: Automatically generate semantic descriptions for MLflow artifacts.



## Build Project

This project uses Poetry for dependency management. To get started, you'll need to install Poetry if you haven't already:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# Use poetry to build the project. Build files are in dist folder
poetry build
```

## Installation

Install from the built filename.gz from the dist folder
```bash
pip dist/install cords_semantics-0.1.0.tar.gz
```


## How to Test

Run the unit tests provided in the tests/ folder
```bash
python3 -m unittest
```

## How to use

CORDS tags collections can be directly imported at MLFlow experiments. These Tags can be set based on the scenario. Following is an example of using CORDS vocabulary tags for loging after an experiment run is finished. 

```python
#Importing CORDS Tags
import cords_semantics.tags as cords_tags


with mlflow.start_run(run_name = run_name) as mlflow_run:
    mlflow_run_id = mlflow_run.info.run_id
    
    mlflow.set_experiment_tag("second best_model", "K-Neighbors Regressor")
    mlflow.set_tag("tag2", "K-Neighbors Regressor")
    mlflow.set_tag(cords_tags.CORDS_RUN, mlflow_run_id)
    mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "K-Neighbors Regressor")
    mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
    mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "sklearn")
    
    mlflow.sklearn.log_model(models['K-Neighbors Regressor'], "knnmodel")
        
    mlflow.log_metric("test_RMSE", rmse_scores.loc[rmse_scores['Model Name'] == 'K-Neighbors Regressor', 'RMSE_Score'].values[0])
    mlflow.log_metric("test_MAE", mae_scores.loc[mae_scores['Model Name'] == 'K-Neighbors Regressor', 'MAE_Score'].values[0])
    mlflow.log_metric("test_R2_Score", r2_scores.loc[r2_scores['Model Name'] == 'K-Neighbors Regressor', 'R2_Score'].values[0])   
    
    mlflow.log_input(dataset, context="training")
    
    
    print("MLFlow Run ID: ", mlflow_run_id)

```

CORDS semantic manager library can be then used to extract and generate the semantic defintion for an experiment.


```python

from cords_semantics.semantics import MlSemanticManager
from cords_semantics.mlflow import extract_mlflow_semantics, convert_tags_to_dictionary

semantic_manager = MlSemanticManager('data/cordsml.rdf')

# use CORDS MLFlow helpers to extract tag values of an experiment run
mflow_tags = extract_mlflow_semantics(mlflow_run_id)
print("tags extracted from mlflow: ", mflow_tags)

mlflow_semantics_dictionary = convert_tags_to_dictionary(mflow_tags)
print("semantic in a dictionary: ", mlflow_semantics_dictionary)

# generating semantic model and serializing it to RDF/XML format
semantic_manager.create_model_semantics(mlflow_semantics_dictionary)

semantic_manager.serialize_model_semantics("/location_of_the_file/lib_output.rdf")

```
