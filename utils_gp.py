from utils import *


def aquisition_function(x, gp, cost_gp, fid_high, gamma, beta, cost_offset):
    cost, cost_var = inference(cost_gp, jnp.array([x]))
    # fixing fidelity
    for i in range(len(fid_high)):
        i += 1
        x = x.at[-i].set(fid_high[-i])
    mean, cov = inference(gp, jnp.array([x]))
    return -((mean[0] + beta * cov[0]) / ((gamma * (cost[0] - cost_offset)))[0])[0]


def greedy_function(x, gp, fid_high):
    # fixing fidelity
    for i in range(len(fid_high)):
        x = jnp.append(x, fid_high[i])
    mean, cov = inference(gp, jnp.array([x]))
    return -mean[0]


def train_gp(inputs, outputs, ms):
    init_params = lhs(
        np.array([[1, 10] for i in range(len(inputs[0, :]))]), ms, log=True
    )
    D = gpx.Dataset(X=inputs, y=outputs)
    for p in init_params:
        best_nll = 1e30
        kern = gpx.Matern52(active_dims=[i for i in range(D.in_dim)])
        prior = gpx.Prior(kernel=kern)
        likelihood = gpx.Gaussian(num_datapoints=D.n)
        posterior = prior * likelihood
        mll = jit(posterior.marginal_log_likelihood(D, negative=True))
        opt = ox.adam(learning_rate=1e-3)

        parameter_state = gpx.initialise(posterior)
        # parameter_state.trainables['likelihood']['obs_noise'] = False
        # parameter_state.params['likelihood']['obs_noise'] = 0.01
        parameter_state.params["kernel"]["lengthscale"] = p

        inference_state = gpx.fit(mll, parameter_state, opt, num_iters=100000)
        nll = float(inference_state.history[-1])
        if nll < best_nll:
            best_nll = nll
            best_inference_state = inference_state
            best_likelihood = likelihood
            best_posterior = posterior

    learned_params, _ = best_inference_state.unpack()
    return best_posterior, learned_params, D, best_likelihood


def inference(gp, inputs):
    posterior = gp["posterior"]
    learned_params = gp["learned_params"]
    D = gp["D"]
    likelihood = gp["likelihood"]
    latent_distribution = posterior(learned_params, D)(inputs)
    predictive_distribution = likelihood(learned_params, latent_distribution)
    predictive_mean = predictive_distribution.mean()
    predictive_cov = predictive_distribution.covariance()
    return predictive_mean, predictive_cov


def build_gp_dict(posterior, learned_params, D, likelihood):
    gp_dict = {}
    gp_dict["posterior"] = posterior
    gp_dict["learned_params"] = learned_params
    gp_dict["D"] = D
    gp_dict["likelihood"] = likelihood
    return gp_dict
