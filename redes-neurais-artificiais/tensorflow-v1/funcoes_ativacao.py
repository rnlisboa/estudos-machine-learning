import numpy as np

# transfer function
def step_function(soma):
    if (soma >= 1):
        return 1
    return 0 

test = step_function(30)

def sigmoide_function(soma):
    return 1 / (1 + np.exp(-soma))

test_2 = sigmoide_function(0.358)
print(test_2)

def tahn_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def relu(soma):
    if(soma >= 0):
        return soma
    return 0

def linear_function(soma):
    return soma

def softmax_function(x):
    ex = np.exp(x)
    return ex / ex.sum()

# print(tahn_function(-0.358))
# print(relu(-0.358))
# print(relu(0.358))
# valores = np.array([5.0, 2.0, 1.3])
# print(softmax_function(valores))

v = 2.1
print(sigmoide_function(v))
print(tahn_function(v))
print(relu(v))
print(linear_function(v))