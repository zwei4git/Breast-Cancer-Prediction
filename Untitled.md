

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

#### Load the data from sklearn


```python
from sklearn.datasets import load_breast_cancer
```


```python
cancer=load_breast_cancer()
```


```python
cancer.data
```




    array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
            1.189e-01],
           [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
            8.902e-02],
           [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
            8.758e-02],
           ...,
           [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
            7.820e-02],
           [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
            1.240e-01],
           [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
            7.039e-02]])



#### Creat the dataframe


```python
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



#### Pairplot to see the pattern


```python
df2=df
```


```python
df2['class']=cancer.target
```


```python
sns.pairplot(df2,hue='class')
```




    <seaborn.axisgrid.PairGrid at 0x1fb53986c50>




![png](output_11_1.png)


### 2class, continuous predictors, try SVM vs DNN


```python
from sklearn.cross_validation import train_test_split
```


```python
X=df
y=cancer.target
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
from sklearn.svm import SVC
```

#### Use default parameters


```python
model=SVC()
```


```python
model.fit(X_train,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
predict=model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(classification_report(y_test,predict))
```

                 precision    recall  f1-score   support
    
              0       0.00      0.00      0.00        66
              1       0.61      1.00      0.76       105
    
    avg / total       0.38      0.61      0.47       171
    
    

    D:\Python\Python\anaconda\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    

#### Grid search to find the best parameters


```python
#Grid search
from sklearn.grid_search import GridSearchCV
```


```python
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.01,0.001,]}
```


```python
grid=GridSearchCV(SVC(),param_grid,verbose=3)
```


```python
grid.fit(X_train,y_train)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=0.1, gamma=1, score=0.631579 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=0.1, gamma=1, score=0.631579 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=0.1, gamma=1, score=0.636364 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.636364 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................... C=0.1, gamma=0.001, score=0.631579 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................... C=0.1, gamma=0.001, score=0.631579 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................... C=0.1, gamma=0.001, score=0.636364 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.631579 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.631579 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.636364 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
    

    [CV] ........................ C=1, gamma=0.01, score=0.636364 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................... C=1, gamma=0.001, score=0.902256 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................... C=1, gamma=0.001, score=0.939850 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................... C=1, gamma=0.001, score=0.954545 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................... C=10, gamma=1, score=0.631579 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................... C=10, gamma=1, score=0.631579 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................... C=10, gamma=1, score=0.636364 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................... C=10, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................... C=10, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................... C=10, gamma=0.01, score=0.636364 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................... C=10, gamma=0.001, score=0.894737 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................... C=10, gamma=0.001, score=0.932331 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................... C=10, gamma=0.001, score=0.916667 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................... C=100, gamma=1, score=0.631579 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................... C=100, gamma=1, score=0.631579 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................... C=100, gamma=1, score=0.636364 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................... C=100, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................... C=100, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................... C=100, gamma=0.01, score=0.636364 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................... C=100, gamma=0.001, score=0.894737 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................... C=100, gamma=0.001, score=0.932331 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................... C=100, gamma=0.001, score=0.916667 -   0.0s
    [CV] C=1000, gamma=1 .................................................
    [CV] ........................ C=1000, gamma=1, score=0.631579 -   0.0s
    [CV] C=1000, gamma=1 .................................................
    [CV] ........................ C=1000, gamma=1, score=0.631579 -   0.0s
    [CV] C=1000, gamma=1 .................................................
    [CV] ........................ C=1000, gamma=1, score=0.636364 -   0.0s
    [CV] C=1000, gamma=0.01 ..............................................
    [CV] ..................... C=1000, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=1000, gamma=0.01 ..............................................
    [CV] ..................... C=1000, gamma=0.01, score=0.631579 -   0.0s
    [CV] C=1000, gamma=0.01 ..............................................
    [CV] ..................... C=1000, gamma=0.01, score=0.636364 -   0.0s
    [CV] C=1000, gamma=0.001 .............................................
    [CV] .................... C=1000, gamma=0.001, score=0.894737 -   0.0s
    [CV] C=1000, gamma=0.001 .............................................
    [CV] .................... C=1000, gamma=0.001, score=0.932331 -   0.0s
    [CV] C=1000, gamma=0.001 .............................................
    [CV] .................... C=1000, gamma=0.001, score=0.916667 -   0.0s
    

    [Parallel(n_jobs=1)]: Done  45 out of  45 | elapsed:    0.5s finished
    




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.01, 0.001]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=3)




```python
grid.best_params_
```




    {'C': 1, 'gamma': 0.001}




```python
grid.best_estimator_
```




    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
grid_pred=grid.predict(X_test)
```


```python
print(classification_report(y_test,grid_pred))
```

                 precision    recall  f1-score   support
    
              0       0.92      0.89      0.91        66
              1       0.93      0.95      0.94       105
    
    avg / total       0.93      0.93      0.93       171
    
    

# DNN 

Standardize the X data


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler=StandardScaler()
```


```python
fit_trans=scaler.fit_transform(df)
```


```python
fit_trans=pd.DataFrame(fit_trans,columns=df.columns)
fit_trans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.097064</td>
      <td>-2.073335</td>
      <td>1.269934</td>
      <td>0.984375</td>
      <td>1.568466</td>
      <td>3.283515</td>
      <td>2.652874</td>
      <td>2.532475</td>
      <td>2.217515</td>
      <td>2.255747</td>
      <td>...</td>
      <td>1.886690</td>
      <td>-1.359293</td>
      <td>2.303601</td>
      <td>2.001237</td>
      <td>1.307686</td>
      <td>2.616665</td>
      <td>2.109526</td>
      <td>2.296076</td>
      <td>2.750622</td>
      <td>1.937015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.829821</td>
      <td>-0.353632</td>
      <td>1.685955</td>
      <td>1.908708</td>
      <td>-0.826962</td>
      <td>-0.487072</td>
      <td>-0.023846</td>
      <td>0.548144</td>
      <td>0.001392</td>
      <td>-0.868652</td>
      <td>...</td>
      <td>1.805927</td>
      <td>-0.369203</td>
      <td>1.535126</td>
      <td>1.890489</td>
      <td>-0.375612</td>
      <td>-0.430444</td>
      <td>-0.146749</td>
      <td>1.087084</td>
      <td>-0.243890</td>
      <td>0.281190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.579888</td>
      <td>0.456187</td>
      <td>1.566503</td>
      <td>1.558884</td>
      <td>0.942210</td>
      <td>1.052926</td>
      <td>1.363478</td>
      <td>2.037231</td>
      <td>0.939685</td>
      <td>-0.398008</td>
      <td>...</td>
      <td>1.511870</td>
      <td>-0.023974</td>
      <td>1.347475</td>
      <td>1.456285</td>
      <td>0.527407</td>
      <td>1.082932</td>
      <td>0.854974</td>
      <td>1.955000</td>
      <td>1.152255</td>
      <td>0.201391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.768909</td>
      <td>0.253732</td>
      <td>-0.592687</td>
      <td>-0.764464</td>
      <td>3.283553</td>
      <td>3.402909</td>
      <td>1.915897</td>
      <td>1.451707</td>
      <td>2.867383</td>
      <td>4.910919</td>
      <td>...</td>
      <td>-0.281464</td>
      <td>0.133984</td>
      <td>-0.249939</td>
      <td>-0.550021</td>
      <td>3.394275</td>
      <td>3.893397</td>
      <td>1.989588</td>
      <td>2.175786</td>
      <td>6.046041</td>
      <td>4.935010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.750297</td>
      <td>-1.151816</td>
      <td>1.776573</td>
      <td>1.826229</td>
      <td>0.280372</td>
      <td>0.539340</td>
      <td>1.371011</td>
      <td>1.428493</td>
      <td>-0.009560</td>
      <td>-0.562450</td>
      <td>...</td>
      <td>1.298575</td>
      <td>-1.466770</td>
      <td>1.338539</td>
      <td>1.220724</td>
      <td>0.220556</td>
      <td>-0.313395</td>
      <td>0.613179</td>
      <td>0.729259</td>
      <td>-0.868353</td>
      <td>-0.397100</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



## Train test split


```python
X=fit_trans
Y=pd.DataFrame(cancer.target,columns=['cancer'])
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
```


```python
import tensorflow as tf
```


```python
import tensorflow.contrib.learn as learn
```


```python
feat_cols = [tf.contrib.layers.real_valued_column("", dimension=len(X.columns))]
```

### 4 hidden layers, with 15, 30, 20, 15 neurons


```python
classifier=learn.DNNClassifier(hidden_units=[15,30,20,15],feature_columns=feat_cols,n_classes=2)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\hp\AppData\Local\Temp\tmppi3vwymd
    INFO:tensorflow:Using config: {'_task_type': None, '_task_id': 0, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000001FB30EFFF60>, '_master': '', '_num_ps_replicas': 0, '_num_worker_replicas': 0, '_environment': 'local', '_is_chief': True, '_evaluation_master': '', '_train_distribute': None, '_eval_distribute': None, '_device_fn': None, '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1.0
    }
    , '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_secs': 600, '_log_step_count_steps': 100, '_protocol': None, '_session_config': None, '_save_checkpoints_steps': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': 'C:\\Users\\hp\\AppData\\Local\\Temp\\tmppi3vwymd'}
    


```python
classifier.fit(x=X_train,y=y_train,steps=200,batch_size=16)
```

    WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
    WARNING:tensorflow:Casting <dtype: 'int32'> labels to bool.
    WARNING:tensorflow:Casting <dtype: 'int32'> labels to bool.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\hp\AppData\Local\Temp\tmppi3vwymd\model.ckpt-200
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 200 into C:\Users\hp\AppData\Local\Temp\tmppi3vwymd\model.ckpt.
    INFO:tensorflow:loss = 0.002811662, step = 201
    INFO:tensorflow:global_step/sec: 597.212
    INFO:tensorflow:loss = 0.069299676, step = 301 (0.169 sec)
    INFO:tensorflow:Saving checkpoints for 400 into C:\Users\hp\AppData\Local\Temp\tmppi3vwymd\model.ckpt.
    INFO:tensorflow:Loss for final step: 0.011860026.
    




    DNNClassifier(params={'head': <tensorflow.contrib.learn.python.learn.estimators.head._BinaryLogisticHead object at 0x000001FB30EFF7F0>, 'hidden_units': [15, 30, 20, 15], 'feature_columns': (_RealValuedColumn(column_name='', dimension=30, default_value=None, dtype=tf.float32, normalizer=None),), 'optimizer': None, 'activation_fn': <function relu at 0x000001FB1EE52EA0>, 'dropout': None, 'gradient_clip_norm': None, 'embedding_lr_multipliers': None, 'input_layer_min_slice_size': None})




```python
pred=classifier.predict(X_test)
```

    WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\hp\AppData\Local\Temp\tmppi3vwymd\model.ckpt-400
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    


```python
from sklearn.metrics import classification_report
```


```python
pred=list(pred)
```


```python
print(classification_report(list(pred),y_test))
```

                 precision    recall  f1-score   support
    
              0       0.95      0.94      0.95        67
              1       0.96      0.97      0.97       104
    
    avg / total       0.96      0.96      0.96       171
    
    
