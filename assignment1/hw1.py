import numpy as np
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def F1(w1,w2):
    a = np.exp(w1)+np.exp(2*w2)
    return np.exp(a)+sigmoid(a)

def F2(w1,w2):
    a = np.exp(w1)+np.exp(2*w2)
    return w1*w2+max(w1,w2)

e = np.exp(1)
print(f'natural number = {e}\n')


#### prob 5(a)
w1 = 1
w2 = -1
f1 = F1(w1,w2)
f2 = F2(w1,w2)
print('prob 5(a)')
print(f'f1 = {f1}')
print(f'f2 = {f2}')
print()

#### prob 5(b)
w1_d = w1 + 0.01
w2_d = w2 + 0.01
f1_dw1 = F1(w1_d,w2)
df1_dw1 = (f1_dw1 - f1)/0.01
print('prob 5(b)')
print(f'df1_dw1 = {df1_dw1}')
f1_dw2 = F1(w1,w2_d)
df1_dw2 = (f1_dw2 - f1)/0.01
print(f'df1_dw2 = {df1_dw2}')
f2_dw1 = F2(w1_d,w2)
df2_dw1 = (f2_dw1 - f2)/0.01
print(f'df2_dw1 = {df2_dw1}')
f2_dw2 = F2(w1,w2_d)
df2_dw2 = (f2_dw2 - f2)/0.01
print(f'df2_dw2 = {df2_dw2}')
print()

### prob 5(c): forward auto-diffirentiation
a = np.exp(w1)+np.exp(2*w2)
df1_dw1 = np.exp(w1)*((sigmoid(a)*(1-sigmoid(a)))+np.exp(a))
print('prob 5(c)')
print(f'df1_dw1 = {df1_dw1}')

df1_dw2 = 2*np.exp(2*w2)*((sigmoid(a)*(1-sigmoid(a)))+np.exp(a))
print(f'df1_dw2 = {df1_dw2}')
print()




