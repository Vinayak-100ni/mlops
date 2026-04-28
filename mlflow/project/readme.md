1.   sudo apt update
2.   sudo apt install python3-pip
3.   sudo apt install python3-venv python3-pip -y
4.   mkdir mlops-project
      cd mlops-project
5.   python3 -m venv mlfow
6.   source mlfow/bin/activate
7.   pip install --upgrade pip
8.   pip install mlflow
9.   pip install dvc
10.  pip install fastapi uvicorn
11.  pip install scikit-learn pandas
12.
```
project/
│
├── data/
├── model/
├── notebooks/
├── src/
├── Dockerfile
├── terraform/
├── k8s/
└── mlruns/
```
13.  install dataset using kaggle and Dataset Setup
```
(mlfow) azureuser@mlops-testing:~/mlops-project/data$ python      
```
```
import kagglehub

path = kagglehub.dataset_download("mosapabdelghany/telcom-customer-churn-dataset")

print("Path:", path)
```
```
 cp -r ~/.cache/kagglehub/datasets/* ~/mlops-project/data
```
14. Data Versioning with DVC
    ```
    dvc init
    git add .
    git commit -m "DVC initialized"
    ```
    ```
    dvc add data/Telco_Cusomer_Churn.csv
    ```
15 . Model Training
```
python src/train.py
```
16. MLflow Tracking
```
mlflow server \
  -h 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:///mnt/mlflow-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```
### Track logs:
learning rate / epochs /max_depth /batch size
```
mlflow.log_param("n_estimators", 100)
```
### Tracks metrices:
accuracy / precision / recall / F1 score /loss
```
mlflow.log_metric("accuracy", 0.89)
```
17. ### Create DVC Pipeline
Create dvc.yaml

#### Pipeline stages:

     Data preparation
     Training
     Evaluation
Run Pipeline :-
```
 dvc repro
```
18..
now the model is created now attact it to backend

create api.py

19. install docker and create dockerfile
    
