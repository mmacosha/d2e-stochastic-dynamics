import torch


def log_normal_density(x, mean, log_var):
    return - 0.5 * (log_var + torch.exp(- log_var) * (mean - x).pow(2))


def compute_fwd_log_likelihood_loss(model, x_t, x_t_m_dt, t, dt):
    r"""
        Compute log p( x_{t} | x_{t-dt} ) with the following logic:
        DRIFT, VAR = model( x_{t-dt}, t-dt )
        log p( x_{t} | x_{t-dt} ) = log N( x_{t} | x_{t-dt} + DRIFT * dt, VAR )
    """
    log_var = torch.as_tensor(2.0 * dt).log()
    
    output = model(x_t_m_dt, t - dt)
    mean_pred = x_t_m_dt + dt * output.drift

    if output.contains('log_var'):
        log_var = output.log_var + log_var
    
    loss = - log_normal_density(x_t, mean_pred, log_var)
    return loss.mean()


def compute_bwd_log_likelihood_loss(model, x_t, x_t_m_dt, t, dt):
    r"""
        Compute log p( x_{t-dt} | x_{t} ) with the following logic:
        DRIFT, VAR = model( x_{t}, t )
        log p( x_{t-dt} | x_{t} ) = log N( x_{t-dt} | x_{t} + DRIFT * dt, VAR )
    """
    log_var = torch.tensor(2.0 * dt).log()

    output = model(x_t, t)
    mean_pred = x_t + dt * output.drift

    if output.contains('log_var'):
        log_var = output.log_var + log_var

    loss = - log_normal_density(x_t_m_dt, mean_pred, log_var)
    return loss.mean()