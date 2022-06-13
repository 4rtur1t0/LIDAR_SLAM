import numpy as np


def maximize_thresholdA(A, v):
    """
    Find a discrete version of v that maximizes w.T*A*w/w.T*w
    """
    # threshs = sorted(v)
    threshs = v
    values = []
    ws = []
    for t in threshs:
        # discretize v to form w
        w = np.zeros(v.size)
        idx = v >= t
        w[idx] = 1
        num = np.dot(w.T, np.dot(A, w))
        den = np.dot(w.T, w)
        values.append(num/den)
        ws.append(w)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(threshs, values)
    plt.show()
    imax = np.argmax(values)
    w = ws[imax]
    return w




def maximize_threshold(v):
    """
    Find a discrete version of v that maximizes w.T*v/|w||v|
    """
    # use the same values in the vector v for thresholding
    # threshs = sorted(v)
    threshs = v
    values = []
    ws = []
    for t in threshs:
        # discretize w
        w = np.zeros(v.size)
        idx = v >= t
        w[idx] = 1
        val = np.dot(w, v)/(np.linalg.norm(w)*np.linalg.norm(v))
        values.append(val)
        ws.append(w)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(threshs, values)
    plt.show()
    imax = np.argmax(values)
    w = ws[imax]
    return w


def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    lam_k = np.dot(b_k.T, np.dot(A, b_k))/np.dot(b_k.T, b_k)
    return lam_k, b_k


def outlier_rejection(A):
    """
    From Recognizing places with spectrally clustered...
    Find a discrete vector w that selects a number of pairwise hypotheses by maximizing the mean probability of
    above all selections. That is, we seek to maximize:
    G = v.T*A*v/(v.T*v)
    Where v_i=1 if the i hypothesis has to be included in the result, else v_i=0.
    First, a vector of real values is computed. To do so, deriving G with respect to v and equalizing to zero yields
    that the vector that maximizes G is one such that:
    A*v = G*v = lambda*v
    that means that we have to find for the eigenvector of the linear application defined by the matrix A.
    In the paper, the power method is used. Here numpy is used along with the np.linalg.eig function.
    The ratio lambda_1/lambda_2 is used to indicate whether the alignment of the hypotheses can be explained in a
    different way. if lambda_1/lambda_2 < 2 then we are safe

    Afterwards, the vector v is discretized to a vector w. The vector w is discretized based on a threshold. The threshold
    is chosen so that it maximizes w.T*v/|w||v|
    """
    k, v = np.linalg.eig(A)
    # sorted indexes
    isrt = np.argsort(k)
    lambda1 = k[isrt[-1]]
    lambda2 = k[isrt[-2]]
    ratio = lambda1/lambda2
    if ratio > 2:
        # if this test is not surpassed, we make no data associations
        return False, np.zeros_like(v)
    # discretize v
    w = maximize_threshold(v)
    return True, w



# # create a row vector of given size
# size = 8
# A = np.random.rand(1, size)
# A = np.dot(A.T, A)
#
# w1, v1 = power_iteration(A, 10)
# w2, v2 = np.linalg.eig(A)
#
# print('Power Method: ', w1, v1)
# print('Eig: ', w2, v2)
#
# w1dis = maximize_threshold(v1)
# w2dis = maximize_thresholdA(A, v1)
#
# print('Discrete and maximum 1: ', w1dis)
# print('Discrete and maximum 2: ', w2dis)