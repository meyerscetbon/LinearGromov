import numpy as np
import time
import scipy
import LinSinkhorn

# Compute GW init
def GW_Init_Cubic(D_1, D_2, a, b):
    P = a[:, None] * b[None, :]
    const_1 = np.dot(
        np.dot(D_1 ** 2, a.reshape(-1, 1)), np.ones(len(b)).reshape(1, -1)
    )  # 2 * n * n + n * m
    const_2 = np.dot(
        np.ones(len(a)).reshape(-1, 1), np.dot(b.reshape(1, -1), (D_2 ** 2).T)
    )  # 2 * m * m + n * m
    const = const_1 + const_2
    L = const - 2 * np.dot(np.dot(D_1, P), D_2)
    res = np.sum(L * P)
    return res


# Compute GW init with factorized cost matrices
def GW_Init(A_1, A_2, B_1, B_2, p, q):
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


### Entropic GW: cubic implementation
## Here the cost considered is C = 2 (constant - 2 DPD')
def GW_entropic_distance(
    D_1,
    D_2,
    reg,
    a,
    b,
    Init="trivial",
    seed_init=49,
    I=10,
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
        X_new = np.sqrt(np.dot(D_1 ** 2, a).reshape(-1, 1))  # 2 * n * n + n
        Y_new = np.sqrt(np.dot(D_2 ** 2, b).reshape(-1, 1))  # 2 * m * m + m
        C_init = LinSinkhorn.Square_Euclidean_Distance(X_new, Y_new)  # n * m
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
        np.dot(D_1 ** 2, a.reshape(-1, 1)), np.ones(len(b)).reshape(1, -1)
    )  # 2 * n * n + n * m
    const_2 = np.dot(
        np.ones(len(a)).reshape(-1, 1), np.dot(b.reshape(1, -1), (D_2 ** 2).T)
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

    for k in range(I):
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


### Entropic GW: quadratic implementation
## Here the cost considered is C = 2 (constant - 2 DPD')
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

        C_init = LinSinkhorn.Square_Euclidean_Distance(X_new, Y_new)
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

    acc.append(res)
    end = time.time()
    curr_time = end - start
    times.append(curr_time)
    list_num_op.append(num_op)

    L = const - 2 * np.dot(
        np.dot(A_1, C_trans), B_2
    )  # n * m + n * d1 * d2 + n * d2 * m
    num_op = num_op + n * m + n * d1 * d2 + n * d2 * m

    for k in range(I):
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

        if np.isnan(res) == True:
            return "Error"
        else:
            acc.append(res)
            Couplings.append(P)

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


## Compute Sinkhorn
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


## Compute Sinkhorn in log scale
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


## LR-GW: quadratic implementation
# If C_init = True, cost is a tuple of matrices (D1,D2)
# If C_init = False, cost is a function
def Quad_LGW_MD(
    X,
    Y,
    a,
    b,
    rank,
    reg,
    alpha,
    cost,
    C_init=False,
    Init="trivial",
    seed_init=49,
    reg_init=1e-1,
    gamma_init="theory",
    gamma_0=1e-1,
    method="IBP",
    max_iter=1000,
    delta=1e-3,
    max_iter_IBP=1000,
    delta_IBP=1e-9,
    lam_IBP=0,
    time_out=200,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []
    Couplings = []

    r = rank
    n, m = np.shape(a)[0], np.shape(b)[0]

    if C_init == True:
        if len(cost) == 2:
            D1, D2 = cost
        else:
            print("Error: cost not adapted")
            return "Error"
    else:
        D1, D2 = cost(X, X), cost(Y, Y)

    ########### Initialization ###########

    ## Init Lower bound
    if Init == "lower_bound":
        X_new = np.sqrt(np.dot(D1 ** 2, a).reshape(-1, 1))  # 2 * n * n + n
        Y_new = np.sqrt(np.dot(D2 ** 2, b).reshape(-1, 1))  # 2 * m * m + m
        C1_init, C2_init = LinSinkhorn.factorized_square_Euclidean(X_new, Y_new)  # 3 * m + 3 * n
        num_op = num_op + 2 * n * n + 2 * m * m + 4 * n + 4 * m
        cost_factorized_init = (C1_init, C2_init)
        cost_init = lambda z1, z2: LinSinkhorn.Square_Euclidean_Distance(z1, z2)

        results = LinSinkhorn.Lin_LOT_MD(
            X_new,
            Y_new,
            a,
            b,
            rank,
            reg,
            alpha,
            cost_init,
            cost_factorized_init,
            Init="kmeans",
            seed_init=seed_init,
            C_init=True,
            reg_init=reg_init,
            gamma_init="arbitrary",
            gamma_0=gamma_0,
            method=method,
            max_iter=max_iter,
            delta=delta_IBP,
            max_iter_IBP=max_iter_IBP,
            delta_IBP=delta_IBP,
            lam_IBP=lam_IBP,
            time_out=100,
        )

        if results == "Error":
            return "Error"

        else:
            res_init, acc_init, times_init, num_op_init, Q, R, g = results
            Couplings.append((Q, R, g))
            num_op = num_op + num_op_init[-1]

    ## Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1  # r
        g = g / np.sum(g)  # r

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1  # n * r
        Q = (Q.T * (a / np.sum(Q, axis=1))).T  # n + 2 * n * r

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
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

    c = np.dot(np.dot(D1 ** 2, a), a) + np.dot(
        np.dot(D2 ** 2, b), b
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

    acc.append(GW_trans)
    end = time.time()
    tim_actual = end - start
    times.append(tim_actual)
    list_num_op.append(num_op)

    err = 1
    niter = 0
    while niter < max_iter:
        Q_prev = Q
        R_prev = R
        g_prev = g
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C2, R)  # r * m * r
            K1_trans_0 = np.dot(C1, K1_trans_0)  # n * r * r
            C1_trans = K1_trans_0 / g + (reg - (1 / gamma)) * np.log(Q)  # 3 * n * r

            K2_trans_0 = np.dot(C1.T, Q)  # r * n * r
            K2_trans_0 = np.dot(C2.T, K2_trans_0)  # m * r * r
            C2_trans = K2_trans_0 / g + (reg - (1 / gamma)) * np.log(R)  # 3 * m * r

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            C3_trans = (omega / (g ** 2)) - (reg - (1 / gamma)) * np.log(g)  # 4 * r

            num_op = (
                num_op + 3 * n * r * r + 2 * m * r * r + 3 * n * r + 3 * m * r + 4 * r
            )

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
                Q, R, g, count_op_IBP = LinSinkhorn.LR_IBP_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

                num_op = num_op + count_op_IBP

            if method == "Dykstra":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
                num_op = num_op + 2 * n * r + 2 * m * r + 2 * r
                Q, R, g, count_op_Dysktra = LinSinkhorn.LR_Dykstra_Sin(
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
            OT_trans = np.trace(G)
            GW_trans = c + OT_trans / 2

            if niter > 10:
                ## Update the error: Practical error
                err = np.abs(GW_trans - acc[-1]) / acc[-1]

                if np.isnan(err):
                    print("Error computation of the stopping criterion", niter)
                    Q = Q_prev
                    R = R_prev
                    g = g_prev
                    break

                # here we let the error to be one always!
                err = 1

            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"

            else:
                acc.append(GW_trans)
                Couplings.append((Q, R, g))
                end = time.time()
                tim_actual = end - start
                times.append(tim_actual)
                list_num_op.append(num_op)
                if tim_actual > time_out:
                    return (
                        acc[-1],
                        np.array(acc),
                        np.array(times),
                        np.array(list_num_op),
                        Couplings,
                    )

        else:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                np.array(list_num_op),
                Couplings,
            )

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Couplings


def update_Quad_cost_GW(D1, D2, Q, R, g):
    n, m = np.shape(D1)[0], np.shape(D2)[0]
    r = np.shape(g)[0]
    cost_trans_1 = np.dot(D1, Q)
    cost_trans_1 = -4 * cost_trans_1 / g
    cost_trans_2 = np.dot(R.T, D2)
    num_op = n * n * r + 2 * n * r + r * m * m
    return cost_trans_1, cost_trans_2, num_op


## LR-GW: linear implementations
# If C_init = True, cost_factorized is a tuple of matrices (D11,D12,D21,D22)
# If C_init = False, cost_factorized is a function
def Lin_LGW_MD(
    X,
    Y,
    a,
    b,
    rank,
    reg,
    alpha,
    cost_factorized,
    C_init=False,
    Init="trivial",
    seed_init=49,
    reg_init=1e-1,
    gamma_init="arbitrary",
    gamma_0=1e-1,
    method="IBP",
    max_iter=1000,
    delta=1e-3,
    max_iter_IBP=1000,
    delta_IBP=1e-9,
    lam_IBP=0,
    time_out=200,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []
    Couplings = []

    if C_init == True:
        if len(cost_factorized) == 4:
            D11, D12, D21, D22 = cost_factorized
        else:
            print("Error: cost not adapted")
            return "Error"
    else:
        D11, D12 = cost_factorized(X, X)
        D21, D22 = cost_factorized(Y, Y)

    r = rank
    n, d1 = np.shape(D11)
    m, d2 = np.shape(D21)

    ########### Initialization ###########

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

        C1_init, C2_init = LinSinkhorn.factorized_square_Euclidean(X_new, Y_new)  # 3 * m + 3 * n
        cost_factorized_init = (C1_init, C2_init)

        num_op = num_op + 4 * n * d1 * d1 + 4 * m * d2 * d2 + 4 * n + 4 * n

        cost_init = lambda z1, z2: LinSinkhorn.Square_Euclidean_Distance(z1, z2)
        results = LinSinkhorn.Lin_LOT_MD(
            X_new,
            Y_new,
            a,
            b,
            rank,
            reg,
            alpha,
            cost_init,
            cost_factorized_init,
            Init="kmeans",
            seed_init=seed_init,
            C_init=True,
            reg_init=reg_init,
            gamma_init="arbitrary",
            gamma_0=gamma_0,
            method=method,
            max_iter=max_iter,
            delta=delta_IBP,
            max_iter_IBP=max_iter_IBP,
            delta_IBP=delta_IBP,
            lam_IBP=lam_IBP,
            time_out=5,
        )

        if results == "Error":
            return "Error"

        else:
            res_init, acc_init, times_init, num_op_init, Q, R, g = results
            Couplings.append((Q, R, g))
            num_op = num_op + num_op_init[-1]

    ## Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1
        g = g / np.sum(g)
        n, d = np.shape(X)
        m, d = np.shape(Y)

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1
        Q = (Q.T * (a / np.sum(Q, axis=1))).T

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
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
        gamma = 1  # to compute

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

    acc.append(GW_trans)
    end = time.time()
    tim_actual = end - start
    times.append(tim_actual)
    list_num_op.append(num_op)

    err = 1
    niter = 0
    while niter < max_iter:
        Q_prev = Q
        R_prev = R
        g_prev = g
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C2, R)  # r * m * r
            K1_trans_0 = np.dot(C1, K1_trans_0)  # n * r * r
            C1_trans = K1_trans_0 / g + (reg - (1 / gamma)) * np.log(Q)  # 3 * n * r

            K2_trans = np.dot(C1.T, Q)  # r * n * r
            K2_trans = np.dot(C2.T, K2_trans)  # m * r * r
            C2_trans = K2_trans / g + (reg - (1 / gamma)) * np.log(R)  # 3 * m * r

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            C3_trans = (omega / (g ** 2)) - (reg - (1 / gamma)) * np.log(g)  # 4 * r

            num_op = (
                num_op + 3 * n * r * r + 2 * m * r * r + 3 * n * r + 3 * m * r + 4 * r
            )

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
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
                K3 = np.exp(gamma * C3_trans)
                num_op = num_op + 2 * n * r + 2 * m * r + 2 * r
                Q, R, g, count_op_Dysktra = LinSinkhorn.LR_Dykstra_Sin(
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
            C1, C2, num_op_update = update_Lin_cost_GW(D11, D12, D21, D22, Q, R, g)
            num_op = num_op + num_op_update

            # GW cost
            C_trans = np.dot(C2, R)
            C_trans = np.dot(C1, C_trans)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)  # \langle -4DPD',P\rangle
            GW_trans = c + OT_trans / 2

            if niter > 10:
                ## Update the error: Practical error
                err = np.abs(GW_trans - acc[-1]) / acc[-1]

                if np.isnan(err):
                    print("Error computation of the stopping criterion", niter)
                    Q = Q_prev
                    R = R_prev
                    g = g_prev
                    break

                # here we let the error to be one always!
                err = 1

            if np.isnan(GW_trans) == True:
                print("Error: NaN GW value")
                return "Error"

            else:
                acc.append(GW_trans)
                Couplings.append((Q, R, g))
                end = time.time()
                tim_actual = end - start
                times.append(tim_actual)
                list_num_op.append(num_op)
                if tim_actual > time_out:
                    return (
                        acc[-1],
                        np.array(acc),
                        np.array(times),
                        np.array(list_num_op),
                        Couplings,
                    )

        else:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                np.array(list_num_op),
                Couplings,
            )

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


## Feature map of k(x,y) = \langle x,y\rangle ** 2
def Feature_Map_Poly(X):
    n, d = np.shape(X)
    X_new = np.zeros((n, d ** 2))
    for i in range(n):
        x = X[i, :][:, None]
        X_new[i, :] = np.dot(x, x.T).reshape(-1)
    return X_new
