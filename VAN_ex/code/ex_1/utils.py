import numpy as np

def read_cameras(datapath):
    with open(datapath + 'calib.txt') as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]

    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)   
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2

def traingulate_point(P, Q, p, q):
    """ 
    traigulating same points from two cameras with SVD
    Args:  
        P ([float 3x4]): matrix of first camera (K[R1|t1])
        Q ([float 3x4]): matrix of second camera (K[R2|t2])
        p ([float 3x1]): pixel coordinate of point on camera 1
        q ([float 3x1]): pixel coordinate of point on camera 2
    """

    P1, P2, P3 = np.vsplit(P, 3)
    Q1, Q2, Q3 = np.vsplit(Q, 3)

    p1, p2 = [p[0], p[1]]
    q1, q2 = [q[0], q[1]]
    
    A = np.vstack([P3 * p1 - P1,
                   P3 * p2 - P2,
                   Q3 * q1 - Q1,
                   Q3 * q2 - Q2])
    
    U, S, VH = np.linalg.svd(A, full_matrices=True)

    # find min singluar value
    minSingularValueIndex = np.argmin(S)
    minSingularValue = S[minSingularValueIndex]
    X = VH[minSingularValueIndex, ...].reshape(-1, 1)
    X = X/ X[-1]
    return X


    

if __name__ == "__main__":
    datapath = "/workspaces/SLAMcourse/VAN_ex/data/dataset05/sequences/05/"
    k, m1, m2 = read_cameras(datapath)
    print("k1=\n{}\n".format(k))
    print("m1=\n{}\n".format(m1))
    print("m2=\n{}\n".format(m2))