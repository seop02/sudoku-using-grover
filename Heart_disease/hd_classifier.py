import gc
import os

import pandas as pd

import torch
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from torch.nn.functional import normalize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import uci_dataset as dataset

def to_torch(X, d_type: torch.dtype):
    m, n = X.shape[0], X.shape[1]
    X_inner = torch.tensor(X, dtype=d_type)

    X_prime = normalize(torch.cat(
        [X_inner, torch.ones(m, dtype=d_type).reshape(-1, 1)], dim=1
    ), p=2, dim=1)
    return X_prime

def kronecker(a: torch.Tensor, b: torch.Tensor):
    return torch.einsum('na,nb->nab', a, b).view(a.size(0), a.size(1) * b.size(1))

if not os.path.exists('./hd_output'):
        os.mkdir('./hd_output')

names = np.zeros(14)
for i in range(14):
    names[i] = i

data = pd.read_csv('cleveland.data', names = names)

data = data.replace("?", np.nan)

label = np.zeros(303)

#raw data had labels: '0', '1', '2', and '3', so converting them into binary classes(0 for healthy, unhealthy otherwise)
for i in range(303):
    if data[13.0][i] == 0:
        label[i] = 0
    else:
        label[i] = 1
        
data['label'] = label

data=data.dropna()
data=data.dropna(axis=0)
data=data.dropna().reset_index(drop=True)
display(data)
data.to_csv(f'./hd_output/raw.csv')

if __name__ == '__main__':
    torch.set_printoptions(linewidth=10000, precision=2, sci_mode=False, threshold=10000)
    torch.set_num_threads(8)

    d_type = torch.float64

    y = data['label'].values
    X = data.drop('label', axis = 1).values
    X = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=539232944)
   
    X_prime = to_torch(X_train, d_type=d_type)
    
    test_class = 0
    test_no = 1
    X_test_prime = to_torch(X_test[y_test == test_class], d_type=d_type)

    max_copies = 100
    result = []
    for copies in range(1, max_copies + 1):
        c = X_test_prime[0]
        result_sum = 0
        result_sum_fid = 0
        for a_index in range(X_prime[y_train == 0].shape[0]):
            for b_index in range(X_prime[y_train == 1].shape[0]):
                a = X_prime[y_train == 0][a_index]
                b = X_prime[y_train == 1][b_index]

                inner_product_ab: torch.Tensor = torch.matmul(a, b)
                inner_product_ca: torch.Tensor = torch.matmul(c, a)
                inner_product_cb: torch.Tensor = torch.matmul(c, b)
                overlap_ab = float(inner_product_ab * torch.conj(inner_product_ab))
                overlap_ca = float(inner_product_ca * torch.conj(inner_product_ca))
                overlap_cb = float(inner_product_cb * torch.conj(inner_product_cb))

                lambda_ab = np.sqrt(1 - np.power(overlap_ab, copies))
                contrib_ca = np.power(overlap_ca, copies)
                contrib_cb = np.power(overlap_cb, copies)

                result_ab = 1/lambda_ab * (contrib_ca - contrib_cb)
                result_ab_fid = contrib_ca - contrib_cb
                result_sum += result_ab
                result_sum_fid += result_ab_fid
                
    
        result_sum = result_sum / (X_prime[y_train == 0].shape[0] * X_prime[y_train == 1].shape[0])
        result_sum_fid = result_sum_fid / (X_prime[y_train == 0].shape[0] * X_prime[y_train == 1].shape[0])
            
        df_copies = pd.DataFrame(
            data=[[copies, result_sum, result_sum_fid]],
            columns=['Copies', 'Helstrom sim', 'Fidelity sim']
        )
        result.append(df_copies)
        df = pd.concat(result)

df.to_csv(f'./hd_output/hd_score_copies_{max_copies}.csv')   

df = pd.read_csv(f'./hd_output/hd_score_copies_{max_copies}.csv')
fig: Figure = plt.figure()
ax: Axes = fig.subplots()
df.plot(x='Copies', y='Helstrom sim', kind='line', ax=ax, label='Helstrom Simulation', color='blue')
df.plot(x='Copies', y='Fidelity sim', kind='line', ax=ax, label='Fidelity Simulation', color='darkred')
ax.set_ylabel('Score')
plt.legend()
plt.savefig(f'./hd_output/score_test_{test_class}_{max_copies}.png')
plt.savefig(f'./hd_output/score_test_{test_class}_{max_copies}.pdf')
plt.show()


#finding confusion matrix for different copies
if __name__ == '__main__':
    torch.set_printoptions(linewidth=10000, precision=2, sci_mode=False, threshold=10000)
    torch.set_num_threads(8)

    d_type = torch.float64


    y = data['label'].values
    X = data.drop('label', axis = 1).values
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X[y != 2], y[y != 2],
                                                        test_size=0.2, random_state=539232944)

    X_prime = to_torch(X_train, d_type=d_type)

    test_no = 10
    X_test_prime = to_torch(X_test, d_type=d_type)

    copies = 2
    result_sum = torch.zeros(X_test_prime.shape[0])
    result_sum_fid = torch.zeros(X_test_prime.shape[0])
    
    for i in range(X_test_prime.shape[0]):
        c = X_test_prime[i] 
        for a_index in range(X_prime[y_train == 0].shape[0]):
            for b_index in range(X_prime[y_train == 1].shape[0]):
                a = X_prime[y_train == 0][a_index]
                b = X_prime[y_train == 1][b_index]

                inner_product_ab: torch.Tensor = torch.matmul(a, b)
                inner_product_ca: torch.Tensor = torch.matmul(c, a)
                inner_product_cb: torch.Tensor = torch.matmul(c, b)
                overlap_ab = float(inner_product_ab * torch.conj(inner_product_ab))
                overlap_ca = float(inner_product_ca * torch.conj(inner_product_ca))
                overlap_cb = float(inner_product_cb * torch.conj(inner_product_cb))

                lambda_ab = np.sqrt(1 - np.power(overlap_ab, copies))
                contrib_ca = np.power(overlap_ca, copies)
                contrib_cb = np.power(overlap_cb, copies)

                result_ab = 1/lambda_ab * (contrib_ca - contrib_cb)
                result_ab_fid = contrib_ca - contrib_cb
                result_sum[i] += result_ab
                result_sum_fid[i] += result_ab_fid
                
    
result_sum = result_sum / (X_prime[y_train == 0].shape[0] * X_prime[y_train == 1].shape[0])
result_sum_fid = result_sum_fid / (X_prime[y_train == 0].shape[0] * X_prime[y_train == 1].shape[0])

df_2 = pd.DataFrame(np.asarray([
            result_sum.numpy().T,
            result_sum_fid.numpy(),
            y_test
        ]).T, columns=['result_sum', 'result_sum_fid', 'y_test'])
df_2.to_csv(f'./hd_output/classification_test_copies_{copies}.csv')


copies = 2
df_2 = pd.read_csv(f'./hd_output/classification_test_copies_{copies}.csv')
result_sum = df_2['result_sum']
result_sum_fid = df_2['result_sum_fid']


hel_pred = np.zeros(X_test_prime.shape[0])
fid_pred = np.zeros(X_test_prime.shape[0])

for i in range(X_test_prime.shape[0]):
    if result_sum[i] > 0:
        hel_pred[i] = 0
    else:
        hel_pred[i] = 1

for i in range(X_test_prime.shape[0]):
    if result_sum_fid[i] > 0:
        fid_pred[i] = 0
    else:
        fid_pred[i] = 1

print("helstrom for copies:{:} \n".format(copies))        
print(metrics.classification_report(y_test, hel_pred, digits=3))
print(metrics.confusion_matrix(y_test, hel_pred))

print("helstrom for copies:{:} \n".format(copies))        
print(metrics.classification_report(y_test, fid_pred, digits=3))
print(metrics.confusion_matrix(y_test, fid_pred))




