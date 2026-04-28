```
The Complete MLOps Flow
Phase 1: Experimentation with MLflow 🧪
Purpose: Find the best model

Train multiple models with different hyperparameters
MLflow logs everything:

Parameters: n_estimators=100, max_depth=10
Metrics: accuracy=0.85, precision=0.82
Models: Saves each trained model


Compare in MLflow UI: See which model performs best
Select winner: Pick the model with highest accuracy/F1-score

Why: You need to experiment and find what works best before production

Phase 2: Production Pipeline with DVC 🏭
Purpose: Make the winning model reproducible

Create DVC pipeline (dvc.yaml) with the best parameters you found
DVC automates:

prepare.py → Clean data
train.py → Train model with best parameters
evaluate.py → Generate metrics


Run dvc repro → Entire pipeline executes automatically
DVC tracks:

Data versions
Code versions
Model versions


Commit to Git: git add dvc.lock → Pipeline state saved

Why: DVC ensures anyone can reproduce your exact model, anytime

Phase 3: Deploy Model in Application 🚀
Purpose: Serve predictions to users

Load the model from model/churn_model.pkl
Create FastAPI endpoint:

python   @app.post('/predict')
   def predict(data):
       prediction = model.predict(data)
       return {'churn': prediction}

Containerize with Docker
Deploy to production (AWS, GCP, Kubernetes)
Your app calls the API: POST /predict → Get predictions

Why: Users need real-time predictions through an API

The Key Difference
ToolWhen to UsePurposeMLflowDuring developmentExperiment, compare, find best modelDVCAfter finding best modelAutomate pipeline, version everythingFastAPI + DockerAfter DVC pipeline worksDeploy model to production

Real Example Flow

(MLflow):

Train 20 different models
Try different algorithms, hyperparameters
MLflow shows Model #15 has best accuracy (89%)


(DVC):

Put Model #15's parameters in train.py
Create dvc.yaml pipeline
Run dvc repro → Reproducible model created
Commit everything to Git


(Deployment):

Create api.py with FastAPI
Build Docker image
Deploy to AWS/GCP
Your mobile app/website calls the API


Future Updates:

New data arrives? → dvc repro retrains automatically
Better model found? → Update pipeline, rebuild Docker
Everything is versioned and traceable
```



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
20. create docker container
    ``` docker build -t churn-model .```
21. run docker container
    ```  docker run -p 8000:8000 churn-model

22. if you change any think in model then do
```
dvc repro
git add dvc.lock reports/metrics.json
git commit -m "New model version"
```
