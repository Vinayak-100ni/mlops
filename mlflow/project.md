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

12.  install dataset using kaggle
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
