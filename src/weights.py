import pandas as pd
import numpy as np


def calculate_bspn(train_df, identity_columns):
    """
    For each identity, calculate whether instances in training data are
    background positive, subgroup negative, background negative, subgroup positive.
    """
    results = {}
    positive = (train_df['target'].values >= 0.5).astype(bool).astype(np.int)
    negative = (train_df['target'].values < 0.5).astype(bool).astype(np.int)
    for i in identity_columns:
        results[i] = dict()
        background = (train_df[i].fillna(0).values < 0.5).astype(bool).astype(np.int)
        subgroup = (train_df[i].fillna(0).values >= 0.5).astype(bool).astype(np.int)
        bp = ((positive + background) > 1).astype(bool).astype(np.int)
        sn = ((negative + subgroup) > 1).astype(bool).astype(np.int)
        bn = ((negative + background) > 1).astype(bool).astype(np.int)
        sp = ((positive + subgroup) > 1).astype(bool).astype(np.int)
        results[i]['bp'] = bp
        results[i]['sn'] = sn
        results[i]['bn'] = bn
        results[i]['sp'] = sp
    return results


def training_weights(train_df, identity_columns, bn_sn_ratio=4):
    """
    Get training weights for each instance in training data.
    Give instances with subgroup negative higher weights.
    """
    r = calculate_bspn(identity_columns)
    bp_identity = pd.DataFrame({i: r[i]['bp'] for i in r})
    sn_identity = pd.DataFrame({i: r[i]['sn'] for i in r})
    bn_identity = pd.DataFrame({i: r[i]['bn'] for i in r})
    sp_identity = pd.DataFrame({i: r[i]['sp'] for i in r})

    if bn_sn_ratio == 4:
        weight_coeff = {'all': 0.2, 'sub': 0.3, 'sn': 0.3, 'bp': 0.3}
        # bn:sn:bp:sp=0.2:0.8:0.5:0.5
    elif bn_sn_ratio == 3:
        weight_coeff = {'all': 0.25, 'sub': 0.25, 'sn': 0.25, 'bp': 0.25}
    elif bn_sn_ratio == 5:
        weight_coeff = {'all': 0.2, 'sub': 0.4, 'sn': 0.4, 'bp': 0.4}
    elif bn_sn_ratio == 6:
        weight_coeff = {'all': 0.2, 'sub': 0.5, 'sn': 0.5, 'bp': 0.5}
    else:
        raise Exception("Ratio not supported.")

    # Overall
    weights = np.ones((train_df.shape[0],), dtype=np.float32) * weight_coeff['all']

    # Subgroup
    weights += (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) * \
               weight_coeff['sub']

    # Subgroup Negative
    weights += sn_identity.mean(axis=1).astype(bool).astype(int) * weight_coeff['sn']

    # Background Positive
    weights += bp_identity.all(axis=1).astype(int) * weight_coeff['bp']

    return weights
