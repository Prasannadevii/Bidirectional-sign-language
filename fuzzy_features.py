import numpy as np
import skfuzzy as fuzz

def compute_fuzzy_features(feature_values):
    x = np.array(feature_values, dtype=np.float32)
    if x.size == 0:
        return np.zeros(11, dtype=np.float32)

    f_min, f_max = x.min(), x.max()
    f_mean, f_std = x.mean(), x.std() if x.std() != 0 else 1e-3

    try:
        trimf = fuzz.trimf(x, [f_min, f_mean, f_max])
        trapmf = fuzz.trapmf(x, [f_min, f_min, f_mean, f_max])
        gaussmf = fuzz.gaussmf(x, f_mean, f_std)
        gauss2mf = fuzz.gauss2mf(x, f_mean-f_std, f_mean+f_std, f_mean-0.5*f_std, f_mean+0.5*f_std)
        gbellmf = fuzz.gbellmf(x, 2*f_std, 2, f_mean)
        sigmf = fuzz.sigmf(x, f_mean, 0.1)
        dsigmf = fuzz.dsigmf(x, f_mean-0.5*f_std, f_mean-0.25*f_std, f_mean+0.25*f_std, f_mean+0.5*f_std)
        psigmf = fuzz.psigmf(x, f_mean-0.5*f_std, f_mean-0.25*f_std, f_mean+0.25*f_std, f_mean+0.5*f_std)
        pimf = fuzz.pimf(x, f_min, f_mean-0.1*f_std, f_mean+0.1*f_std, f_max)
        smf = fuzz.smf(x, f_min, f_mean)
        zmf = fuzz.zmf(x, f_mean, f_max)
    except:
        return np.zeros(11, dtype=np.float32)

    fuzzy_features = np.array([np.mean(trimf), np.mean(trapmf), np.mean(gaussmf),
                               np.mean(gauss2mf), np.mean(gbellmf), np.mean(sigmf),
                               np.mean(dsigmf), np.mean(psigmf), np.mean(pimf),
                               np.mean(smf), np.mean(zmf)], dtype=np.float32)

    return np.nan_to_num(fuzzy_features)
