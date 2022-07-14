import numpy as np
import time
import LinSinkhorn
import utils
from sklearn.cluster import KMeans
from sklearn import preprocessing


def KL(A, B):
    Ratio_trans = np.log(A) - np.log(B)
    return np.sum(A * Ratio_trans)


# D1 = A_1A_2 and D2 = B_1B_2
def GW_init_factorized(A_1, A_2, B_1, B_2, p, q):
    tilde_A_1 = Feature_Map_Poly(A_1)
    tilde_A_2_T = Feature_Map_Poly(A_2.T)
    tilde_A_2 = tilde_A_2_T.T

    tilde_B_1 = Feature_Map_Poly(B_1)
    tilde_B_2_T = Feature_Map_Poly(B_2.T)
    tilde_B_2 = tilde_B_2_T.T

    tilde_a = np.dot(tilde_A_1, np.dot(tilde_A_2, p))
    tilde_b = np.dot(tilde_B_1, np.dot(tilde_B_2, q))

    c = np.dot(tilde_a, p) + np.dot(tilde_b, q)

    P1 = p[:, None]
    P2 = q[None, :]
    G_1 = np.dot(A_2, P1)
    G_2 = np.dot(P2, B_1)
    G = np.dot(G_1, G_2)
    G_1_1 = np.dot(B_2, P2.T)
    G_2_1 = np.dot(P1.T, A_1)
    G_trans = np.dot(G_1_1, G_2_1)

    M = np.dot(G, G_trans)
    res = c - 2 * np.trace(M)
    return res


# need
def GW_init_cubic(D_1, D_2, a, b):
    P = a[:, None] * b[None, :]
    const_1 = np.dot(
        np.dot(D_1**2, a.reshape(-1, 1)), np.ones(len(b)).reshape(1, -1)
    )  # 2 * n * n + n * m
    const_2 = np.dot(
        np.ones(len(a)).reshape(-1, 1), np.dot(b.reshape(1, -1), (D_2**2).T)
    )  # 2 * m * m + n * m
    const = const_1 + const_2
    L = const - 2 * np.dot(np.dot(D_1, P), D_2)
    res = np.sum(L * P)
    return res


#### CUBIC VERSION ####
## Stable version: works for every $\varepsilon ##
# Here the costs considered are C = 2 (constant - 2 DPD')
def GW_entropic_distance(
    D_1,
    D_2,
    reg,
    a,
    b,
    Init="trivial",
    seed_init=49,
    I=10,
    delta=1e-6,
    delta_sin=1e-9,
    num_iter_sin=1000,
    lam_sin=0,
    LSE=False,
    time_out=50,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []
    Couplings = []

    n, m = np.shape(a)[0], np.shape(b)[0]

    if Init == "trivial":
        P = a[:, None] * b[None, :]
        Couplings.append(P)
        num_op = num_op + n * m

    if Init == "lower_bound":
        X_new = np.sqrt(np.dot(D_1**2, a).reshape(-1, 1))  # 2 * n * n + n
        Y_new = np.sqrt(np.dot(D_2**2, b).reshape(-1, 1))  # 2 * m * m + m
        C_init = Square_Euclidean_Distance(X_new, Y_new)  # n * m
        num_op = num_op + n * m + 2 * n * n + 2 * m * m + n + m

        if LSE == False:
            u, v, K, count_op_Sin = Sinkhorn(
                C_init, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin
            P = u[:, None] * K * v[None, :]
            num_op = num_op + 2 * n * m
        else:
            P, count_op_Sin_LSE = LSE_Sinkhorn(
                C_init, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin_LSE

        Couplings.append(P)

    if Init == "random":
        np.random.seed(seed_init)
        P = np.abs(np.random.randn(n, m))
        P = P + 1
        P = (P.T * (a / np.sum(P, axis=1))).T
        Couplings.append(P)
        num_op = num_op + 3 * n * m + n

    const_1 = np.dot(
        np.dot(D_1**2, a.reshape(-1, 1)), np.ones(len(b)).reshape(1, -1)
    )  # 2 * n * n + n * m
    const_2 = np.dot(
        np.ones(len(a)).reshape(-1, 1), np.dot(b.reshape(1, -1), (D_2**2).T)
    )  # 2 * m * m + n * m
    num_op = num_op + 2 * n * m + 2 * n * n + 2 * m * m
    const = const_1 + const_2
    L = const - 2 * np.dot(np.dot(D_1, P), D_2)

    res = np.sum(L * P)
    # print(res)
    end = time.time()
    curr_time = end - start
    times.append(curr_time)
    acc.append(res)
    list_num_op.append(num_op)

    err = 1
    for k in range(I):
        if err < delta:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                np.array(list_num_op),
                Couplings,
            )
        P_prev = P
        if LSE == False:
            u, v, K, count_op_Sin = Sinkhorn(
                2 * L, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin
            P = u.reshape((-1, 1)) * K * v.reshape((1, -1))
            num_op = num_op + 2 * n * m
        else:
            P, count_op_Sin_LSE = LSE_Sinkhorn(
                2 * L, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin_LSE

        L = const - 2 * np.dot(np.dot(D_1, P), D_2)
        num_op = num_op + n * n * m + n * m * m + 2 * n * m
        res = np.sum(L * P)
        # print(res)

        if np.isnan(res) == True:
            return "Error"
        else:
            acc.append(res)
            Couplings.append(P)

        err = np.linalg.norm(P - P_prev)
        end = time.time()
        curr_time = end - start
        times.append(curr_time)
        list_num_op.append(num_op)
        if curr_time > time_out:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                np.array(list_num_op),
                Couplings,
            )

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Couplings


#### QUAD VERSION ####
## Stable version: works for every $\varepsilon ##
# Here the costs considered are C = 2 (constant - 2 DPD')
def Quad_GW_entropic_distance(
    A_1,
    A_2,
    B_1,
    B_2,
    reg,
    a,
    b,
    Init="trivial",
    seed_init=49,
    I=10,
    delta=1e-6,
    delta_sin=1e-9,
    num_iter_sin=1000,
    lam_sin=0,
    time_out=50,
    LSE=False,
):
    start = time.time()
    num_op = 0

    acc = []
    times = []
    list_num_op = []
    Couplings = []

    n, d1 = np.shape(A_1)
    m, d2 = np.shape(B_1)

    tilde_A_1 = Feature_Map_Poly(A_1)
    tilde_A_2_T = Feature_Map_Poly(A_2.T)
    tilde_A_2 = tilde_A_2_T.T

    tilde_B_1 = Feature_Map_Poly(B_1)
    tilde_B_2_T = Feature_Map_Poly(B_2.T)
    tilde_B_2 = tilde_B_2_T.T

    num_op = num_op + 2 * n * d1 * d1 + 2 * m * d2 * d2

    tilde_a = np.dot(tilde_A_1, np.dot(tilde_A_2, a))  # 2 * d1 * d1 * n
    tilde_b = np.dot(tilde_B_1, np.dot(tilde_B_2, b))  # 2 * d2 * d2 * m

    c = np.dot(tilde_a, a) + np.dot(tilde_b, b)  # n + m

    const_1 = np.dot(tilde_a.reshape(-1, 1), np.ones(len(b)).reshape(1, -1))  # n * m
    const_2 = np.dot(np.ones(len(a)).reshape(-1, 1), tilde_b.reshape(1, -1))  # n * m
    const = const_1 + const_2

    num_op = num_op + 2 * d1 * d1 * n + 2 * d2 * d2 * m + 3 * n * m

    if Init == "trivial":
        P = a[:, None] * b[None, :]
        Couplings.append(P)
        num_op = num_op + n * m

    if Init == "lower_bound":
        X_new = np.dot(tilde_A_2, a)
        X_new = np.sqrt(np.dot(tilde_A_1, X_new).reshape(-1, 1))
        Y_new = np.dot(tilde_B_2, b)
        Y_new = np.sqrt(np.dot(tilde_B_1, Y_new).reshape(-1, 1))

        C_init = Square_Euclidean_Distance(X_new, Y_new)
        num_op = num_op + n * m + 2 * d1 * d1 * n + 2 * d2 * d2 * m + n + m

        if LSE == False:
            u, v, K, count_op_Sin = Sinkhorn(
                C_init, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin
            P = u[:, None] * K * v[None, :]
            num_op = num_op + 2 * n * m
        else:
            P, count_op_Sin_LSE = LSE_Sinkhorn(
                C_init, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin_LSE

        Couplings.append(P)

    if Init == "random":
        np.random.seed(seed_init)
        P = np.abs(np.random.randn(n, m))
        P = P + 1
        P = (P.T * (a / np.sum(P, axis=1))).T
        Couplings.append(P)
        num_op = num_op + 3 * n * m + n

    C_trans = np.dot(np.dot(A_2, P), B_1)  # d1 * n * m + d1 * m * d2
    num_op = num_op + d1 * n * m + d1 * d2 * m

    C_trans_2 = np.dot(np.dot(B_2, P.T), A_1)
    C_f = np.dot(C_trans_2, C_trans)
    res = c - 2 * np.trace(C_f)
    # print(res)

    acc.append(res)
    end = time.time()
    curr_time = end - start
    times.append(curr_time)
    list_num_op.append(num_op)

    L = const - 2 * np.dot(
        np.dot(A_1, C_trans), B_2
    )  # n * m + n * d1 * d2 + n * d2 * m
    num_op = num_op + n * m + n * d1 * d2 + n * d2 * m

    err = 1
    for k in range(I):
        if err < delta:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                np.array(list_num_op),
                Couplings,
            )
        P_prev = P
        if LSE == False:
            u, v, K, count_op_Sin = Sinkhorn(
                2 * L, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            P = u.reshape((-1, 1)) * K * v.reshape((1, -1))
            num_op = num_op + count_op_Sin + 2 * n * m
        else:
            P, count_op_Sin_LSE = LSE_Sinkhorn(
                2 * L, reg, a, b, delta=delta_sin, num_iter=num_iter_sin, lam=lam_sin
            )
            num_op = num_op + count_op_Sin_LSE

        C_trans = np.dot(np.dot(A_2, P), B_1)
        L = const - 2 * np.dot(np.dot(A_1, C_trans), B_2)
        num_op = num_op + d1 * n * m + d2 * n * m + d1 * d2 * n + d1 * d2 * m + n * m

        C_trans_2 = np.dot(np.dot(B_2, P.T), A_1)
        C_f = np.dot(C_trans_2, C_trans)
        res = c - 2 * np.trace(C_f)
        # print(res)

        if np.isnan(res) == True:
            return "Error"
        else:
            acc.append(res)
            Couplings.append(P)

        err = np.linalg.norm(P - P_prev)
        end = time.time()
        curr_time = end - start
        times.append(curr_time)
        list_num_op.append(num_op)
        if curr_time > time_out:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                np.array(list_num_op),
                Couplings,
            )

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Couplings


#######  GROMOV WASSERSTEIN #######
def update_Quad_cost_GW(D1, D2, Q, R, g):
    n, m = np.shape(D1)[0], np.shape(D2)[0]
    r = np.shape(g)[0]
    cost_trans_1 = np.dot(D1, Q)
    cost_trans_1 = -4 * cost_trans_1 / g
    cost_trans_2 = np.dot(R.T, D2)
    num_op = n * n * r + 2 * n * r + r * m * m
    return cost_trans_1, cost_trans_2, num_op


# If C_init = True, cost is a tuple of matrices
# If C_init = False, cost is a function
# Init = 'trivial', 'random', 'lower_bound'
def Quad_LGW_MD(
    X,
    Y,
    a,
    b,
    rank,
    cost,
    time_out=200,
    max_iter=1000,
    delta=1e-3,
    gamma_0=10,
    gamma_init="rescale",
    reg=0,
    alpha=1e-10,
    C_init=False,
    Init="lower_bound",
    seed_init=49,
    reg_init=1e-1,
    method="Dykstra",
    max_iter_IBP=10000,
    delta_IBP=1e-3,
    lam_IBP=0,
    rescale_cost=True,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []
    Couplings = []

    if gamma_0 * reg >= 1:
        # display(Latex(f'Choose $\gamma$ and $\epsilon$ such that $\gamma$ x $\epsilon<1$'))
        print("gamma et epsilon must be well choosen")
        return "Error"

    r = rank
    n, m = np.shape(a)[0], np.shape(b)[0]

    if C_init == True:
        if len(cost) == 2:
            D1, D2 = cost
            if rescale_cost == True:
                D1, D2 = D1 / D1.max(), D2 / D2.max()
        else:
            print("Error: cost not adapted")
            return "Error"
    else:
        D1, D2 = cost(X, X), cost(Y, Y)
        if rescale_cost == True:
            D1, D2 = D1 / D1.max(), D2 / D2.max()

    ########### Initialization ###########
    if Init == "kmeans":
        g = np.ones(rank) / rank
        kmeans_X = KMeans(n_clusters=rank, random_state=0).fit(X)
        num_iter_kmeans_X = kmeans_X.n_iter_
        Z_X = kmeans_X.cluster_centers_
        C_trans_X = utils.Square_Euclidean_Distance(X, Z_X)
        C_trans_X = C_trans_X / C_trans_X.max()
        results = utils.Sinkhorn(
            C_trans_X,
            reg_init,
            a,
            g,
            max_iter=max_iter_IBP,
            delta=delta_IBP,
            lam=lam_IBP,
            time_out=1e100,
        )
        res, arr_acc_X, arr_times_X, Q, arr_num_op_X = results

        # lb_X = preprocessing.LabelBinarizer()
        # lb_X.fit(kmeans_X.labels_)
        # Q = lb_X.transform(kmeans_X.labels_)
        # Q = (Q.T * a).T

        kmeans_Y = KMeans(n_clusters=rank, random_state=0).fit(Y)
        num_iter_kmeans_Y = kmeans_Y.n_iter_
        Z_Y = kmeans_Y.cluster_centers_
        C_trans_Y = utils.Square_Euclidean_Distance(Y, Z_Y)
        C_trans_Y = C_trans_Y / C_trans_Y.max()
        results = utils.Sinkhorn(
            C_trans_Y,
            reg_init,
            b,
            g,
            max_iter=max_iter_IBP,
            delta=delta_IBP,
            lam=lam_IBP,
            time_out=1e100,
        )
        res, arr_acc_Y, arr_times_Y, R, arr_num_op_Y = results

        # lb_Y = preprocessing.LabelBinarizer()
        # lb_Y.fit(kmeans_Y.labels_)
        # R = lb_Y.transform(kmeans_Y.labels_)
        # R = (R.T * b).T

        num_op = (
            num_op
            + (num_iter_kmeans_X + np.shape(arr_acc_X)[0]) * rank * np.shape(X)[0]
            + (num_iter_kmeans_Y + np.shape(arr_acc_Y)[0]) * rank * np.shape(Y)[0]
        )

    ## Init Lower bound
    if Init == "lower_bound":
        X_new = np.sqrt(np.dot(D1**2, a).reshape(-1, 1))  # 2 * n * n + n
        Y_new = np.sqrt(np.dot(D2**2, b).reshape(-1, 1))  # 2 * m * m + m
        num_op = num_op + 2 * n * n + 2 * m * m
        cost_factorized_init = lambda X, Y: factorized_square_Euclidean(X, Y)
        cost_init = lambda z1, z2: Square_Euclidean_Distance(z1, z2)

        results = LinSinkhorn.Lin_LOT_MD(
            X_new,
            Y_new,
            a,
            b,
            rank,
            cost_init,
            cost_factorized_init,
            reg=0,
            alpha=1e-10,
            gamma_0=gamma_0,
            max_iter=1000,
            delta=1e-3,
            time_out=5,
            Init="kmeans",
            seed_init=49,
            C_init=False,
            reg_init=1e-1,
            gamma_init="rescale",
            method="Dykstra",
            max_iter_IBP=10000,
            delta_IBP=1e-3,
            lam_IBP=0,
            rescale_cost=True,
        )

        res_init, acc_init, times_init, num_op_init, list_criterion, Q, R, g = results
        Couplings.append((Q, R, g))
        num_op = num_op + num_op_init[-1]

        # print('res: '+str(res_init))

    ## Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1  # r
        g = g / np.sum(g)  # r

        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1  # n * r
        Q = (Q.T * (a / np.sum(Q, axis=1))).T  # n + 2 * n * r

        R = np.abs(np.random.randn(m, rank))
        R = R + 1  # n * r
        R = (R.T * (b / np.sum(R, axis=1))).T  # m + 2 * m * r

        Couplings.append((Q, R, g))
        num_op = num_op + 2 * n * r + 2 * m * r + n + m + 2 * r

    ## Init trivial
    if Init == "trivial":
        g = np.ones(rank) / rank
        lambda_1 = min(np.min(a), np.min(g), np.min(b)) / 2

        a1 = np.arange(1, np.shape(a)[0] + 1)
        a1 = a1 / np.sum(a1)
        a2 = (a - lambda_1 * a1) / (1 - lambda_1)

        b1 = np.arange(1, np.shape(b)[0] + 1)
        b1 = b1 / np.sum(b1)
        b2 = (b - lambda_1 * b1) / (1 - lambda_1)

        g1 = np.arange(1, rank + 1)
        g1 = g1 / np.sum(g1)
        g2 = (g - lambda_1 * g1) / (1 - lambda_1)

        Q = lambda_1 * np.dot(a1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            a2[:, None], g2.reshape(1, -1)
        )
        R = lambda_1 * np.dot(b1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            b2[:, None], g2.reshape(1, -1)
        )

        Couplings.append((Q, R, g))
        num_op = num_op + 4 * n * r + 4 * m * r + 3 * n + 3 * m + 3 * r
    #####################################

    if gamma_init == "theory":
        gamma = 1  # to compute

    if gamma_init == "regularization":
        gamma = 1 / reg

    if gamma_init == "arbitrary":
        gamma = gamma_0

    c = np.dot(np.dot(D1**2, a), a) + np.dot(
        np.dot(D2**2, b), b
    )  # 2 * n * n + n + 2 * m * m + m
    C1, C2, num_op_update = update_Quad_cost_GW(D1, D2, Q, R, g)
    num_op = num_op + 2 * n * n + n + 2 * m * m + m + num_op_update

    # GW cost
    C_trans = np.dot(C2, R)
    C_trans = np.dot(C1, C_trans)
    C_trans = C_trans / g
    G = np.dot(Q.T, C_trans)
    OT_trans = np.trace(G)  # \langle -4DPD',P\rangle
    GW_trans = c + OT_trans / 2
    # print(GW_trans)

    acc.append(GW_trans)
    end = time.time()
    time_actual = end - start
    times.append(time_actual)
    list_num_op.append(num_op)

    err = 1
    niter = 0
    count_escape = 1
    while (niter < max_iter) and (time_actual < time_out):
        Q_prev = Q
        R_prev = R
        g_prev = g
        # P_prev = np.dot(Q/g,R.T)
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C2, R)  # r * m * r
            K1_trans_0 = np.dot(C1, K1_trans_0)  # n * r * r
            grad_Q = K1_trans_0 / g + reg * np.log(Q)
            if gamma_init == "rescale":
                # norm_1 = np.linalg.norm(grad_Q)**2
                norm_1 = np.max(np.abs(grad_Q)) ** 2

            K2_trans_0 = np.dot(C1.T, Q)  # r * n * r
            K2_trans_0 = np.dot(C2.T, K2_trans_0)  # m * r * r
            grad_R = K2_trans_0 / g + reg * np.log(R)
            if gamma_init == "rescale":
                # norm_2 = np.linalg.norm(grad_R)**2
                norm_2 = np.max(np.abs(grad_R)) ** 2

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            grad_g = -(omega / (g**2)) + reg * np.log(g)
            if gamma_init == "rescale":
                # norm_3 = np.linalg.norm(grad_g)**2
                norm_3 = np.max(np.abs(grad_g)) ** 2

            if gamma_init == "rescale":
                gamma = gamma_0 / max(norm_1, norm_2, norm_3)

            C1_trans = grad_Q - (1 / gamma) * np.log(Q)  # 3 * n * r
            C2_trans = grad_R - (1 / gamma) * np.log(R)  # 3 * m * r
            C3_trans = grad_g - (1 / gamma) * np.log(g)  # 4 * r

            num_op = (
                num_op + 3 * n * r * r + 2 * m * r * r + 3 * n * r + 3 * m * r + 4 * r
            )

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp((-gamma) * C3_trans)
                Q, R, g = LinSinkhorn.LR_IBP_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            if method == "Dykstra":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp((-gamma) * C3_trans)
                num_op = num_op + 2 * n * r + 2 * m * r + 2 * r
                Q, R, g, count_op_Dysktra, n_iter_Dykstra = LinSinkhorn.LR_Dykstra_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    alpha,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

                num_op = num_op + count_op_Dysktra

            if method == "Dykstra_LSE":
                Q, R, g, count_op_Dysktra_LSE = LinSinkhorn.LR_Dykstra_LSE_Sin(
                    C1_trans,
                    C2_trans,
                    C3_trans,
                    a,
                    b,
                    alpha,
                    gamma,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

                num_op = num_op + count_op_Dysktra_LSE

            # Update the total cost
            C1, C2, num_op_update = update_Quad_cost_GW(D1, D2, Q, R, g)
            num_op = num_op + num_op_update

            # GW cost
            C_trans = np.dot(C2, R)
            C_trans = np.dot(C1, C_trans)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)  # \langle -4DPD',P\rangle
            GW_trans = c + OT_trans / 2
            # print(GW_trans)

            if np.isnan(GW_trans) == True:
                print("Error LR-GW: GW cost", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            ## Update the error: Practical error
            # err = np.abs(GW_trans - acc[-1]) / acc[-1]
            # err = np.abs(GW_trans - acc[-1]) / np.log(num_op - list_num_op[-1])

            ## Update error: difference between couplings
            # P_act = np.dot(Q/g,R.T)
            # err = np.linalg.norm(P_act - P_prev)
            # print(err)

            ## Update the error: theoritical error
            err_1 = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
            err_2 = ((1 / gamma) ** 2) * (KL(R, R_prev) + KL(R_prev, R))
            err_3 = ((1 / gamma) ** 2) * (KL(g, g_prev) + KL(g_prev, g))
            criterion = err_1 + err_2 + err_3
            # print(criterion)

            if niter > 1:
                if criterion > delta / 1e-1:
                    err = criterion
                else:
                    count_escape = count_escape + 1
                    if count_escape != niter:
                        err = criterion

            if np.isnan(criterion) == True:
                print("Error LR-GW: stopping criterion", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            acc.append(GW_trans)
            Couplings.append((Q, R, g))
            time_actual = time.time() - start
            times.append(time_actual)
            list_num_op.append(num_op)

        else:
            break

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Couplings


def update_Lin_cost_GW(D11, D12, D21, D22, Q, R, g):
    n, d1 = np.shape(D11)
    m, d2 = np.shape(D21)
    r = np.shape(g)[0]
    cost_trans_1 = np.dot(D12, Q)  # d1 * n * r
    cost_trans_1 = -4 * np.dot(
        D11, cost_trans_1 / g
    )  # n * d1 * r + d1 * r + n * r # size: n * r
    cost_trans_2 = np.dot(R.T, D21)  # r * m * d2
    cost_trans_2 = np.dot(cost_trans_2, D22)  # r * d2 * m # size: r * m
    num_op = 2 * n * r * d1 + 2 * r * d2 * m + d1 * r + n * r
    return cost_trans_1, cost_trans_2, num_op


# If C_init = True, cost_factorized is a tuple of matrices (D11,D12,D21,D22)
# D1 = D11D12, D2 = D21D22
# If C_init = False, cost_factorized is a function
# Init = 'trivial', 'random', 'lower_bound'
def Lin_LGW_MD(
    X,
    Y,
    a,
    b,
    rank,
    cost_factorized,
    time_out=50,
    max_iter=1000,
    delta=1e-3,
    gamma_0=10,
    gamma_init="rescale",
    reg=0,
    alpha=1e-10,
    C_init=False,
    Init="lower_bound",
    seed_init=49,
    reg_init=1e-1,
    method="Dykstra",
    max_iter_IBP=10000,
    delta_IBP=1e-3,
    lam_IBP=0,
    rescale_cost=True,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []
    Couplings = []
    list_niter_Dykstra = []

    if gamma_0 * reg >= 1:
        # display(Latex(f'Choose $\gamma$ and $\epsilon$ such that $\gamma$ x $\epsilon<1$'))
        print("gamma et epsilon must be well choosen")
        return "Error"

    if C_init == True:
        if len(cost_factorized) == 4:
            D11, D12, D21, D22 = cost_factorized
            if rescale_cost == True:
                D11, D12, D21, D22 = (
                    D11 / np.sqrt(np.max(D11)),
                    D12 / np.sqrt(np.max(D12)),
                    D21 / np.sqrt(np.max(D21)),
                    D22 / np.sqrt(np.max(D22)),
                )
        else:
            print("Error: cost not adapted")
            return "Error"
    else:
        D11, D12 = cost_factorized(X, X)
        D21, D22 = cost_factorized(Y, Y)
        if rescale_cost == True:
            D11, D12, D21, D22 = (
                D11 / np.sqrt(np.max(D11)),
                D12 / np.sqrt(np.max(D12)),
                D21 / np.sqrt(np.max(D21)),
                D22 / np.sqrt(np.max(D22)),
            )

    r = rank
    n, d1 = np.shape(D11)
    m, d2 = np.shape(D21)
    ########### Initialization ###########
    if Init == "kmeans":
        g = np.ones(rank) / rank
        kmeans_X = KMeans(n_clusters=rank, random_state=0).fit(X)
        num_iter_kmeans_X = kmeans_X.n_iter_
        Z_X = kmeans_X.cluster_centers_
        C_trans_X = utils.Square_Euclidean_Distance(X, Z_X)
        C_trans_X = C_trans_X / C_trans_X.max()
        results = utils.Sinkhorn(
            C_trans_X,
            reg_init,
            a,
            g,
            max_iter=max_iter_IBP,
            delta=delta_IBP,
            lam=lam_IBP,
            time_out=1e100,
        )
        res, arr_acc_X, arr_times_X, Q, arr_num_op_X = results

        # lb_X = preprocessing.LabelBinarizer()
        # lb_X.fit(kmeans_X.labels_)
        # Q = lb_X.transform(kmeans_X.labels_)
        # Q = (Q.T * a).T

        kmeans_Y = KMeans(n_clusters=rank, random_state=0).fit(Y)
        num_iter_kmeans_Y = kmeans_Y.n_iter_
        Z_Y = kmeans_Y.cluster_centers_
        C_trans_Y = utils.Square_Euclidean_Distance(Y, Z_Y)
        C_trans_Y = C_trans_Y / C_trans_Y.max()
        results = utils.Sinkhorn(
            C_trans_Y,
            reg_init,
            b,
            g,
            max_iter=max_iter_IBP,
            delta=delta_IBP,
            lam=lam_IBP,
            time_out=1e100,
        )
        res, arr_acc_Y, arr_times_Y, R, arr_num_op_Y = results

        # lb_Y = preprocessing.LabelBinarizer()
        # lb_Y.fit(kmeans_Y.labels_)
        # R = lb_Y.transform(kmeans_Y.labels_)
        # R = (R.T * b).T

        num_op = (
            num_op
            + (num_iter_kmeans_X + np.shape(arr_acc_X)[0]) * rank * np.shape(X)[0]
            + (num_iter_kmeans_Y + np.shape(arr_acc_Y)[0]) * rank * np.shape(Y)[0]
        )

    ## Init Lower bound
    if Init == "lower_bound":
        tilde_D11 = Feature_Map_Poly(D11)  # n * d1 * d1
        tilde_D12_T = Feature_Map_Poly(D12.T)  # n * d1 * d1
        tilde_D12 = tilde_D12_T.T

        tilde_D21 = Feature_Map_Poly(D21)  # m * d2 * d2
        tilde_D22_T = Feature_Map_Poly(D22.T)  # m * d2 * d2
        tilde_D22 = tilde_D22_T.T

        X_new = np.dot(tilde_D12, a)  # d1 * d1 * n
        X_new = np.sqrt(np.dot(tilde_D11, X_new).reshape(-1, 1))  # n * d1 * d1 + n
        Y_new = np.dot(tilde_D22, b)  # d2 * d2 * m
        Y_new = np.sqrt(np.dot(tilde_D21, Y_new).reshape(-1, 1))  # m * d2 * d2 + m

        num_op = num_op + 4 * n * d1 * d1 + 4 * m * d2 * d2 + 4 * n + 4 * n

        cost_factorized_init = lambda X, Y: factorized_square_Euclidean(
            X, Y
        )  # 3 * m + 3 * n
        cost_init = lambda z1, z2: Square_Euclidean_Distance(z1, z2)
        results = LinSinkhorn.Lin_LOT_MD(
            X_new,
            Y_new,
            a,
            b,
            rank,
            cost_init,
            cost_factorized_init,
            reg=0,
            alpha=1e-10,
            gamma_0=gamma_0,
            max_iter=1000,
            delta=1e-3,
            time_out=5,
            Init="kmeans",
            seed_init=49,
            C_init=False,
            reg_init=1e-1,
            gamma_init="rescale",
            method="Dykstra",
            max_iter_IBP=10000,
            delta_IBP=1e-3,
            lam_IBP=0,
            rescale_cost=True,
        )

        (
            res_init,
            acc_init,
            times_init,
            num_op_init,
            list_criterion_init,
            Q,
            R,
            g,
        ) = results
        Couplings.append((Q, R, g))
        num_op = num_op + num_op_init[-1]
        # print('res: '+str(res_init))

    ## Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1
        g = g / np.sum(g)
        n, d = np.shape(X)
        m, d = np.shape(Y)

        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1
        Q = (Q.T * (a / np.sum(Q, axis=1))).T

        R = np.abs(np.random.randn(m, rank))
        R = R + 1
        R = (R.T * (b / np.sum(R, axis=1))).T

        Couplings.append((Q, R, g))
        num_op = num_op + 2 * n * r + 2 * m * r + n + m + 2 * r

    ## Init trivial
    if Init == "trivial":
        g = np.ones(rank) / rank
        lambda_1 = min(np.min(a), np.min(g), np.min(b)) / 2

        a1 = np.arange(1, np.shape(a)[0] + 1)
        a1 = a1 / np.sum(a1)
        a2 = (a - lambda_1 * a1) / (1 - lambda_1)

        b1 = np.arange(1, np.shape(b)[0] + 1)
        b1 = b1 / np.sum(b1)
        b2 = (b - lambda_1 * b1) / (1 - lambda_1)

        g1 = np.arange(1, rank + 1)
        g1 = g1 / np.sum(g1)
        g2 = (g - lambda_1 * g1) / (1 - lambda_1)

        Q = lambda_1 * np.dot(a1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            a2[:, None], g2.reshape(1, -1)
        )
        R = lambda_1 * np.dot(b1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            b2[:, None], g2.reshape(1, -1)
        )

        Couplings.append((Q, R, g))
        num_op = num_op + 4 * n * r + 4 * m * r + 3 * n + 3 * m + 3 * r
    #####################################

    if gamma_init == "theory":
        gamma = 1

    if gamma_init == "regularization":
        gamma = 1 / reg

    if gamma_init == "arbitrary":
        gamma = gamma_0

    tilde_D11 = Feature_Map_Poly(D11)  # n * d1 * d1
    tilde_D12_T = Feature_Map_Poly(D12.T)  # n * d1 * d1
    tilde_D12 = tilde_D12_T.T

    tilde_D21 = Feature_Map_Poly(D21)  # m * d2 * d2
    tilde_D22_T = Feature_Map_Poly(D22.T)  # m * d2 * d2
    tilde_D22 = tilde_D22_T.T

    a_tilde = np.dot(
        np.dot(tilde_D12, a), np.dot(np.transpose(tilde_D11), a)
    )  # 2 * d1 * d1 * n + d1 * d1
    b_tilde = np.dot(
        np.dot(tilde_D22, b), np.dot(np.transpose(tilde_D21), b)
    )  # 2 * m * d2 * d2 + d2 * d2
    c = a_tilde + b_tilde
    num_op = num_op + 4 * n * d1 * d1 + 4 * m * d2 * d2 + d1 * d1 + d2 * d2

    C1, C2, num_op_update = update_Lin_cost_GW(D11, D12, D21, D22, Q, R, g)
    num_op = num_op + num_op_update

    C_trans = np.dot(C2, R)
    C_trans = np.dot(C1, C_trans)
    C_trans = C_trans / g
    G = np.dot(Q.T, C_trans)
    OT_trans = np.trace(G)  # \langle -4DPD',P\rangle
    GW_trans = c + OT_trans / 2
    # print(GW_trans)

    acc.append(GW_trans)
    end = time.time()
    time_actual = end - start
    times.append(time_actual)
    list_num_op.append(num_op)

    err = 1
    niter = 0
    count_escape = 1
    while (niter < max_iter) and (time_actual < time_out):
        Q_prev = Q
        R_prev = R
        g_prev = g
        # P_prev = np.dot(Q/g,R.T)
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C2, R)  # d * m * r
            K1_trans_0 = np.dot(C1, K1_trans_0)  # n * d * r
            grad_Q = K1_trans_0 / g
            if reg != 0.0:
                grad_Q = grad_Q + reg * np.log(Q)

            if gamma_init == "rescale":
                norm_1 = np.max(np.abs(grad_Q)) ** 2

            K2_trans_0 = np.dot(C1.T, Q)  # d * n * r
            K2_trans_0 = np.dot(C2.T, K2_trans_0)  # m * d * r
            grad_R = K2_trans_0 / g
            if reg != 0.0:
                grad_R = grad_R + reg * np.log(R)
            if gamma_init == "rescale":
                norm_2 = np.max(np.abs(grad_R)) ** 2

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            grad_g = -omega / (g**2)
            if reg != 0.0:
                grad_g = grad_g + reg * np.log(g)
            if gamma_init == "rescale":
                norm_3 = np.max(np.abs(grad_g)) ** 2

            if gamma_init == "rescale":
                gamma = gamma_0 / max(norm_1, norm_2, norm_3)

            C1_trans = grad_Q - (1 / gamma) * np.log(Q)  # 3 * n * r
            C2_trans = grad_R - (1 / gamma) * np.log(R)  # 3 * m * r
            C3_trans = grad_g - (1 / gamma) * np.log(g)  # 4 * r

            num_op = (
                num_op + 3 * n * r * r + 2 * m * r * r + 3 * n * r + 3 * m * r + 4 * r
            )

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp((-gamma) * C3_trans)
                Q, R, g = LinSinkhorn.LR_IBP_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            if method == "Dykstra":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp((-gamma) * C3_trans)
                num_op = num_op + 2 * n * r + 2 * m * r + 2 * r
                Q, R, g, count_op_Dysktra, n_iter_Dykstra = LinSinkhorn.LR_Dykstra_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    alpha,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

                num_op = num_op + count_op_Dysktra
                list_niter_Dykstra.append(n_iter_Dykstra)

            if method == "Dykstra_LSE":
                Q, R, g, count_op_Dysktra_LSE = LinSinkhorn.LR_Dykstra_LSE_Sin(
                    C1_trans,
                    C2_trans,
                    C3_trans,
                    a,
                    b,
                    alpha,
                    gamma,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

                num_op = num_op + count_op_Dysktra_LSE
            # Update the total cost
            C1, C2, num_op_update = update_Lin_cost_GW(D11, D12, D21, D22, Q, R, g)
            num_op = num_op + num_op_update

            # GW cost
            C_trans = np.dot(C2, R)
            C_trans = np.dot(C1, C_trans)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)  # \langle -4DPD',P\rangle
            GW_trans = c + OT_trans / 2
            # print(GW_trans)

            if np.isnan(GW_trans) == True:
                print("Error LR-GW: GW cost", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            ## Update the error: theoritical error
            # err_1 = ((1/gamma)**2) * (KL(Q,Q_prev) + KL(Q_prev,Q))
            # err_2 = ((1/gamma)**2) * (KL(R,R_prev) + KL(R_prev,R))
            # err_3 = ((1/gamma)**2) * (KL(g,g_prev) + KL(g_prev,g))
            # err = err_1 + err_2 + err_3

            ## Update the error: Practical error
            # err = np.abs(GW_trans - acc[-1]) / acc[-1]
            # err = np.abs(GW_trans - acc[-1]) / np.log(num_op - list_num_op[-1])

            ## Update error: difference between couplings
            # P_act = np.dot(Q/g,R.T)
            # err = np.linalg.norm(P_act - P_prev)
            # print(err)

            err_1 = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
            err_2 = ((1 / gamma) ** 2) * (KL(R, R_prev) + KL(R_prev, R))
            err_3 = ((1 / gamma) ** 2) * (KL(g, g_prev) + KL(g_prev, g))
            criterion = err_1 + err_2 + err_3
            # print(criterion)

            if niter > 1:
                if criterion > delta / 1e-1:
                    err = criterion
                else:
                    count_escape = count_escape + 1
                    if count_escape != niter:
                        err = criterion

            if np.isnan(criterion) == True:
                print("Error LR-GW: stopping criterion", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            # here we let the error to be one always !
            # err = 1

            acc.append(GW_trans)
            Couplings.append((Q, R, g))
            end = time.time()
            time_actual = end - start
            times.append(time_actual)
            list_num_op.append(num_op)

        else:
            break

    return (
        acc[-1],
        np.array(acc),
        np.array(times),
        np.array(list_num_op),
        Couplings,
        np.array(list_niter_Dykstra),
    )


def Sinkhorn(C, reg, a, b, delta=1e-9, num_iter=1000, lam=1e-6):

    n, m = np.shape(C)
    # K = np.exp(-C/reg)
    # Next 3 lines equivalent to K= np.exp(-C/reg), but faster to compute
    K = np.empty(C.shape, dtype=C.dtype)
    np.divide(C, -reg, out=K)  # n * m
    np.exp(K, out=K)  # n * m

    u = np.ones(np.shape(a)[0])  # /np.shape(a)[0]
    v = np.ones(np.shape(b)[0])  # /np.shape(b)[0]

    v_trans = np.dot(K.T, u) + lam  # add regularization to avoid divide 0

    err = 1
    index = 0
    while index < num_iter:
        uprev = u
        vprev = v
        if err > delta:
            index = index + 1

            v = b / v_trans

            u_trans = np.dot(K, v) + lam  # add regularization to avoid divide 0
            u = a / u_trans

            if (
                np.any(np.isnan(u))
                or np.any(np.isnan(v))
                or np.any(np.isinf(u))
                or np.any(np.isinf(v))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", index)
                u = uprev
                v = vprev
                break

            v_trans = np.dot(K.T, u) + lam
            err = np.sum(np.abs(v * v_trans - b))

        else:
            num_op = 3 * n * m + (index + 1) * (2 * n * m + n + m)
            return u, v, K, num_op

    num_op = 3 * n * m + (index + 1) * (2 * n * m + n + m)
    return u, v, K, num_op


def LSE_Sinkhorn(C, reg, a, b, num_iter=1000, delta=1e-3, lam=0):

    f = np.zeros(np.shape(a)[0])
    g = np.zeros(np.shape(b)[0])

    n, m = np.shape(C)

    C_tilde = f[:, None] + g[None, :] - C  # 2 * n * m
    C_tilde = C_tilde / reg  # n * m
    P = np.exp(C_tilde)

    err = 1
    n_iter = 0
    while n_iter < num_iter:
        P_prev = P
        if err > delta:
            n_iter = n_iter + 1

            # Update f
            f = reg * np.log(a) + f - reg * scipy.special.logsumexp(C_tilde, axis=1)

            # Update g
            C_tilde = f[:, None] + g[None, :] - C
            C_tilde = C_tilde / reg
            g = reg * np.log(b) + g - reg * scipy.special.logsumexp(C_tilde, axis=0)

            if (
                np.any(np.isnan(f))
                or np.any(np.isnan(g))
                or np.any(np.isinf(f))
                or np.any(np.isinf(g))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                P = P_prev
                break

            # Update the error
            C_tilde = f[:, None] + g[None, :] - C
            C_tilde = C_tilde / reg
            P = np.exp(C_tilde)
            err = np.sum(np.abs(np.sum(P, axis=1) - a))

        else:
            num_op = 4 * n * m + (n_iter + 1) * (8 * n * m + 3 * n + 3 * m) + n * m
            return P, num_op

    num_op = 4 * n * m + (n_iter + 1) * (8 * n * m + 3 * n + 3 * m) + n * m
    return P, num_op


## Feature map of k(x,y) = \langle x,y\rangle ** 2 ##
def Feature_Map_Poly(X):
    n, d = np.shape(X)
    X_new = np.zeros((n, d**2))
    for i in range(n):
        x = X[i, :][:, None]
        X_new[i, :] = np.dot(x, x.T).reshape(-1)
    return X_new


def Square_Euclidean_Distance(X, Y):
    """Returns the matrix of $|x_i-y_j|^2$."""
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    # D = (np.sum(X ** 2, 1)[:, np.newaxis] - 2 * np.dot(X, Y.T) + np.sum(Y ** 2, 1))
    return C


# shape of xs: num_samples * dimension
def factorized_square_Euclidean(xs, xt):
    square_norm_s = np.sum(xs**2, axis=1)
    square_norm_t = np.sum(xt**2, axis=1)
    A_1 = np.zeros((np.shape(xs)[0], 2 + np.shape(xs)[1]))
    A_1[:, 0] = square_norm_s
    A_1[:, 1] = np.ones(np.shape(xs)[0])
    A_1[:, 2:] = -2 * xs

    A_2 = np.zeros((2 + np.shape(xs)[1], np.shape(xt)[0]))
    A_2[0, :] = np.ones(np.shape(xt)[0])
    A_2[1, :] = square_norm_t
    A_2[2:, :] = xt.T

    return A_1, A_2
