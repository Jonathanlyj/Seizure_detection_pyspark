# Distributed Computation of Seizure Detection Program 

The program investigates an alternative approach to deploy a seizure detection model with distributed computing using PySpark. The platform that the project based off with is GCP.The problem context and data resource is [here](https://www.kaggle.com/c/seizure-detection)

## Dependency requirement
 * python 3.6.5
 * gcsfs==0.2.1
 * numpy==1.16.3
 * scikit-learn==0.21.0
 * scipy==1.2.1
 
 ## General steps to run notebook on GCP DataProc
1. Set up GCP projects and bucket. Store the repo in the bucket associated with this project
2. Modify SETTING.json file to update the following info accordingly:
    "gcp-project-name": gcp project name
    "gcp-bucket-project-dir": directory from this bucket to the repo
    "data-cache-dir": folder name that saves trained models
    "dataset-dir": folder name where dataset is stored
    "submission-dir": folder name stores prediction result for test data
3. Upload seizure datasets to "dataset-dir" folder
4. Launch a cluster using dataproc and establish a ssh channel to the master node machine. Remember to specify driver executor memory for the cluster
5. Use local browser to open jupyter notebooks with PySpark kernal. Open a terminal page first and install following packages:

```sh
pip install scipy
pip install sklearn
pip install gcsfs
```
6. Open CrossValidation, ModelTraining, Predict ipython files and speficy following variable
```py
num_nodes = 4  # number of worker nodes of the cluster
subjects = ['Patient_8']# list of subjects to process 
gs_dir = "gs://seizure_detection_data/notebooks/seizure_detection_spark_gcp"#repo dir on gcp bucket
```
7. Run CrossValidation ipython file to tune model parameters based on validation result
8. Run ModelTraining ipython file to train model with full scope of training data and save trained model
9. Run Predict ipython file to predict test data and generate prediction result csv files.




## Documentation

