import sb.nn as sbnn
from sb import metrics


def test_path_kl():
    class DummyModel:
        def __init__(self, drift_fn):
            self.drift_fn = drift_fn

        def __call__(self, x, t):
            drift = self.drift_fn(x)
            return sbnn.utils.ModelOutput(drift=drift, log_var=0) 

    var = 2
    alpha = 0.4
    dt = 0.01
    n_steps = 20
    t_max = int(n_steps * dt)
    x0 = torch.randn(512, 2)
    
    # Test ll matching
    ll_drift_fn = lambda x: - alpha * x
    kl = metrics.compute_path_kl(
        DummyModel(ll_drift_fn), 
        x0, dt, t_max, n_steps, alpha, var, 
        method="ll"
    )
    assert abs(kl) < 1e-6, "LL KL not close to 0"

    # Test score matching
    score_drift_fn = lambda x: - alpha * x * dt
    kl = metrics.compute_path_kl(
        DummyModel(score_drift_fn), 
        x0, dt, t_max, n_steps, alpha, var, 
        method="score"
    )
    assert abs(kl) < 1e-6, "Score KL not close to 0"

    # Test mean matching
    mean_drift_fn = lambda x: x - alpha * x * dt
    kl = metrics.compute_path_kl(
        DummyModel(mean_drift_fn), 
        x0, dt, t_max, n_steps, alpha, var, 
        method="mean"
    )
    assert abs(kl) < 1e-6, "Mean KL not close to 0"
