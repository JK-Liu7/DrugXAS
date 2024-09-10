import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy import sparse


def feature_cos(data, target, t):
    features = data['feature_dict'][target]
    cos_graph = cos(features)

    for i in range(len(cos_graph)):
        for j in range(len(cos_graph)):
            if cos_graph[i][j] >= t:
                cos_graph[i][j] = 1
            else:
                cos_graph[i][j] = 0

    cos_graph = sparse.csr_matrix(cos_graph)
    return cos_graph


def mp_data(X_train_p, args):
    d = args.drug_number
    e = args.entity_number
    label = np.array([1] * len(X_train_p), dtype=int)

    dr_ent = np.concatenate((X_train_p, np.expand_dims(label, axis=1)), axis=1)
    dr_ent = sp.coo_matrix((np.ones(dr_ent.shape[0]),(dr_ent[:, 0], dr_ent[:, 1])), shape=(d, e)).toarray()
    ded = np.matmul(dr_ent, dr_ent.T)
    ded = sp.coo_matrix(ded)

    ent_dr = np.concatenate((X_train_p[:, (1, 0)], np.expand_dims(label, axis=1)), axis=1)
    ent_dr = sp.coo_matrix((np.ones(ent_dr.shape[0]), (ent_dr[:, 0], ent_dr[:, 1])), shape=(e, d)).toarray()
    ede = np.matmul(ent_dr, ent_dr.T)
    ede = sp.coo_matrix(ede)
    return ded, ede


def mp_pos(ded, ede, drug_sim, entity_sim, args, data):

    d = args.drug_number
    e = args.entity_number
    pos_num = args.pos_num

    # drug
    ded = ded.A.astype("float32")
    dia_d = sp.dia_matrix((np.ones(d), 0), shape=(d, d)).toarray()
    dd = np.ones((d, d)) - dia_d
    ded += drug_sim
    ded = ded * dd
    pos_d_mp = np.zeros((d, d))
    for i in range(d):
        pos_d_mp[i, i] = 1
        one = ded[i].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-ded[i, one])
            sele = one[oo[:pos_num - 1]]
            pos_d_mp[i, sele] = 1
        else:
            pos_d_mp[i, one] = 1
    pos_d = sp.coo_matrix(pos_d_mp)

    # entity
    ede = ede.A.astype("float32")
    dia_e = sp.dia_matrix((np.ones(e), 0), shape=(e, e)).toarray()
    ee = np.ones((e, e)) - dia_e
    ede += entity_sim
    ede = ede * ee
    pos_e_mp = np.zeros((e, e))
    for j in range(e):
        pos_e_mp[j, j] = 1
        one = ede[j].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-ede[j, one])
            sele = one[oo[:pos_num - 1]]
            pos_e_mp[j, sele] = 1
        else:
            pos_e_mp[j, one] = 1
    pos_e = sp.coo_matrix(pos_e_mp)
    return pos_d, pos_e

