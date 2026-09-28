import numpy as np


def softmax(arr):
    s = []
    sum = 0
    for x in arr:
        sum = sum + np.exp(x)

    for x in arr:
        s.append(np.exp(x)/sum)

    return np.array(s)


def derivative(arr):
    d = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i == j:
                d.append(arr[i]*(1-arr[i]))
            else:
                d.append(-arr[i]*arr[j])

    return np.array(d)

if __name__ == "__main__":
    x = np.array([1.12, 2.34, 1.78, 9.1, 1.23, 2.14])/np.sqrt(512)
    s = softmax(x)
    print(f"after softmax: {s}")
    print(f"after derivtive:{derivative(s)}")
