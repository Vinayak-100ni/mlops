Code cell <-F6KloejAQ1W>
# %% [code]
import numpy as np
import pandas as pd


Code cell <8_fa7SocAwh6>
# %% [code]
df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

Code cell <5mJgGT4mBx4C>
# %% [code]
df.shape

Code cell <pQN7h4UdCIvS>
# %% [code]
df.dropna(inplace=True)
df.shape

Code cell <6DpIzS3JCZbK>
# %% [code]
df.duplicated().sum()

Code cell <xDOzBwJ_CfwD>
# %% [code]
df.drop_duplicates(inplace=True)

Code cell <FNq1tkeoC26T>
# %% [code]
df['clean_comment'] = df['clean_comment'].str.lower()
df.head()

Code cell <-CzWmzg6Dg7c>
# %% [code]
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df , x='category')


Code cell <1mcSAWvylmHN>
# %% [code]
!pip install mlflow


Code cell <IHHuwo6Nlwnl>
# %% [code]
import mlflow

mlflow.set_tracking_uri("http://20.220.34.109:5000/")

with mlflow.start_run():
  mlflow.log_param("param1", 15)
  mlflow.log_metric("metric1" , 0.89)

Code cell <jSCIn488Izq3>
# %% [code]
import numpy as np
import pandas as pd

Code cell <AAGh5uVPI64e>
# %% [code]
df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

Code cell <l7nT3iVBJFmf>
# %% [code]
df.dropna(inplace=True)
df.shape

Code cell <xoVTnf3vJRSF>
# %% [code]
df.drop_duplicates(inplace=True)

Code cell <KK2Zh-cHJYdV>
# %% [code]
df =  df[~(df['clean_comment'].str.strip() == '')]
df.shape

Code cell <fuLuIlSAJst9>
# %% [code]
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

Code cell <CVt-fnKVKvId>
# %% [code]
nltk.download('stopwords')
nltk.download('wordnet')

Code cell <KzjAsKf0K21V>
# %% [code]
def preprocess_comment(comment):
  comment = comment.lower()
  comment = comment.strip()
  comment = re.sub(r'\n', ' ' , comment)
  return comment

Code cell <nunztmrjL1OW>
# %% [code]
df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
df.head()

Code cell <Nbggvr4_MG-F>
# %% [code]
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Code cell <ckDpcoKVM6pe>
# %% [code]
import pandas as pd
import matplotlib.pyplot as ptl
import seaborn as sns

Code cell <dFR827ldNi1e>
# %% [code]
vectorize = CountVectorizer(max_features=10000)

Code cell <tjY239bLN2N1>
# %% [code]
X = vectorize.fit_transform(df['clean_comment']).toarray()
y = df['category']

Code cell <D6JRBQcYN9lG>
# %% [code]
X


Code cell <2QvDXw4lOQPe>
# %% [code]
X.shape

Code cell <QsdKPRgyOUIm>
# %% [code]
y



Code cell <hpQhKER5OdHG>
# %% [code]
y.shape


Code cell <xuzOEn62OvIl>
# %% [code]
mlflow.set_tracking_uri('http://20.220.34.109:5000/')

Code cell <lzSrYWoSPBs2>
# %% [code]
mlflow.set_experiment("RF Baseline")

Code cell <BOc33GNRQ0vd>
# %% [code]



