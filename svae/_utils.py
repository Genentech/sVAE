import torch


class GumbelSigmoid(torch.nn.Module):
    def __init__(self, num_action, num_latent, freeze=False, drawhard=True, tau=1):
        super(GumbelSigmoid, self).__init__()
        self.shape = (num_action, num_latent)
        self.freeze = freeze
        self.drawhard = drawhard
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.shape))
        self.tau = tau
        # useful to make sure these parameters will be pushed to the GPU
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.register_buffer("fixed_mask", torch.ones(self.shape))
        self.reset_parameters()

    # changed this to draw one action per minibatch sample...
    def forward(self, action):
        bs = action.shape[0]
        if self.freeze:
            y = self.fixed_mask[action, :]
            return y
        else:
            shape = tuple([bs] + [self.shape[1]])
            logistic_noise = (
                self.sample_logistic(shape)
                .type(self.log_alpha.type())
                .to(self.log_alpha.device)
            )
            y_soft = torch.sigmoid((self.log_alpha[action] + logistic_noise) / self.tau)

            if self.drawhard:
                y_hard = (y_soft > 0.5).type(y_soft.type())

                # This weird line does two things:
                #   1) at forward, we get a hard sample.
                #   2) at backward, we differentiate the gumbel sigmoid
                y = y_hard.detach() - y_soft.detach() + y_soft

            else:
                y = y_soft

            return y

    def get_proba(self):
        """Returns probability of getting one"""
        if self.freeze:
            return self.fixed_mask
        else:
            return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(
            self.log_alpha, 5
        )  # 5)  # will yield a probability ~0.99. Inspired by DCDI

    def sample_logistic(self, shape):
        u = self.uniform.sample(shape)
        return torch.log(u) - torch.log(1 - u)

    def threshold(self):
        proba = self.get_proba()
        self.fixed_mask.copy_((proba > 0.5).type(proba.type()))
        self.freeze = True
