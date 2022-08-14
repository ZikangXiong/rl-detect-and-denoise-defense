import numpy as np
import torch as th
from torch import nn

from neural_shield.config import default_device, defense_config


class GRUVAE(nn.Module):

    def __init__(self, env_config, running_mean=None, running_std=None):
        super(GRUVAE, self).__init__()

        gru_vae_hyper_param = env_config["gru_vae_hyper_param"]
        input_shape = env_config["detector"]["obs_shape"]

        latent_shape = gru_vae_hyper_param["latent_shape"]
        enc_hindden_size = gru_vae_hyper_param["encoder"]["hidden_size"]
        dec_hindden_size = gru_vae_hyper_param["decoder"]["hidden_size"]

        self.encoder = nn.GRU(input_shape, **gru_vae_hyper_param["encoder"])
        self.decoder = nn.GRU(latent_shape, **gru_vae_hyper_param["decoder"])

        self.z_mean_layer = nn.Linear(enc_hindden_size, latent_shape)
        self.z_log_var_layer = nn.Linear(enc_hindden_size, latent_shape)

        self.out_mean_layer = nn.Linear(dec_hindden_size, input_shape)
        self.out_log_var_layer = nn.Linear(dec_hindden_size, input_shape)

        self.enc_hn = None
        self.dec_hn = None

        if running_mean is None or running_std is None:
            self.running_mean = nn.Parameter(th.zeros(input_shape, dtype=th.float32,
                                                      device=default_device), requires_grad=False)
            self.running_std = nn.Parameter(th.ones(input_shape, dtype=th.float32,
                                                    device=default_device), requires_grad=False)
        else:
            self.running_mean = nn.Parameter(th.tensor(running_mean, dtype=th.float32,
                                                       device=default_device), requires_grad=False)
            self.running_std = nn.Parameter(th.tensor(running_std, dtype=th.float32,
                                                      device=default_device), requires_grad=False)

        self.device = default_device
        self.to(self.device)

    def reset(self):
        self.enc_hn = None
        self.dec_hn = None

    def normalize(self, x):
        return (x - self.running_mean) / (self.running_std + 1e-5)

    def denormalize(self, x):
        return (x * self.running_std) + self.running_mean

    def encode(self, x):
        if len(x.shape) == 1 or len(x.shape) == 2:
            x = x.view([1, -1, x.shape[-1]])

        out, enc_hn = self.encoder(x, self.enc_hn)

        z_mean = self.z_mean_layer(out)
        z_log_var = self.z_log_var_layer(out)
        z_reparam_sample = th.normal(mean=th.zeros_like(z_mean, device=self.device),
                                     std=th.ones_like(z_log_var, device=self.device))
        z = z_mean + z_log_var.exp().sqrt() * z_reparam_sample

        return (z, z_mean, z_log_var), enc_hn

    def decode(self, z):
        out, dec_hn = self.decoder(z, self.dec_hn)

        out_mean = self.out_mean_layer(out)
        out_log_var = self.out_log_var_layer(out)
        reparam_sample = th.normal(mean=th.zeros_like(out_mean, device=self.device),
                                   std=th.ones_like(out_log_var, device=self.device))
        out = out_mean + out_log_var.exp().sqrt() * reparam_sample

        return (out, out_mean, out_log_var), dec_hn

    def forward(self, x, stateful=False, deterministic_z=False):
        norm_x = self.normalize(x)

        if stateful:
            (z, z_mean, _), self.enc_hn = self.encode(norm_x)
            self.enc_hn = self.enc_hn.detach()
        else:
            (z, z_mean, _), _ = self.encode(norm_x)

        if deterministic_z:
            z = z_mean

        if stateful:
            (out, _, _), self.dec_hn = self.decode(z)
            self.dec_hn = self.dec_hn.detach()
        else:
            (out, _, _), _ = self.decode(z)

        return self.denormalize(out)


class Detector(GRUVAE):
    def __init__(self, env_config, algo, running_mean=None, running_std=None):
        super(Detector, self).__init__(env_config, running_mean, running_std)
        self.detector_thr = env_config["detector"]["detector_thr"][algo]

    def loss(self, x: th.Tensor, beta: float = 1.0):
        """loss function
        """
        inp = self.normalize(x)
        expect_out = self.normalize(x)

        (z, _, z_log_var), _ = self.encode(inp)
        (out, _, _), _ = self.decode(z)

        recon_loss = th.mean(th.abs(out - expect_out))
        # replace the KL loss with simple regression loss on the z_log_var
        reg_loss = th.mean(th.abs(z_log_var))
        loss = recon_loss + beta * reg_loss

        return {"loss": loss, "recon_loss": recon_loss, "reg_loss": reg_loss}

    def detect(self, obs):
        obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
        norm_obs = self.normalize(obs_tensor)

        pred_obs = self.forward(obs_tensor, stateful=True, deterministic_z=False)
        norm_pred_obs = self.normalize(pred_obs)

        if th.mean(th.abs(norm_pred_obs - norm_obs)) > self.detector_thr:
            return True
        return False


class GRURepairer(GRUVAE):
    def loss(self, adv_x: th.Tensor, x: th.Tensor, beta: float = 0.1):
        """loss function
        """
        inp = self.normalize(adv_x)
        expect_out = self.normalize(x)

        (z, _, z_log_var), _ = self.encode(inp)
        (out, _, _), _ = self.decode(z)

        recon_loss = th.mean(th.abs(out - expect_out))
        reg_loss = th.mean(th.abs(z_log_var))

        loss = recon_loss + beta * reg_loss

        return {"loss": loss, "recon_loss": recon_loss, "reg_loss": reg_loss}

    def repair(self, obs):
        obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
        pred_obs = self.forward(obs_tensor, stateful=True, deterministic_z=False)

        return pred_obs.detach().cpu().numpy().flatten()


class MLPRepairer(nn.Module):
    def __init__(self, env_config, running_mean=None, running_std=None):
        super(MLPRepairer, self).__init__()

        input_shape = env_config["detector"]["obs_shape"]
        self.enc1 = nn.Linear(input_shape, 128)
        self.enc2 = nn.Linear(128, 128)
        self.dec1 = nn.Linear(128, 128)
        self.dec2 = nn.Linear(128, input_shape)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        if running_mean is None or running_std is None:
            self.running_mean = nn.Parameter(th.zeros(input_shape, dtype=th.float32,
                                                      device=default_device), requires_grad=False)
            self.running_std = nn.Parameter(th.ones(input_shape, dtype=th.float32,
                                                    device=default_device), requires_grad=False)
        else:
            self.running_mean = nn.Parameter(th.tensor(running_mean, dtype=th.float32,
                                                       device=default_device), requires_grad=False)
            self.running_std = nn.Parameter(th.tensor(running_std, dtype=th.float32,
                                                      device=default_device), requires_grad=False)

        self.device = default_device
        self.to(self.device)

    def normalize(self, x):
        return (x - self.running_mean) / (self.running_std + 1e-5)

    def denormalize(self, x):
        return (x * (self.running_std + 1e-5)) + self.running_mean

    def forward(self, x):
        norm_x = self.normalize(x)
        # encoder
        h = self.enc1(norm_x)
        h = self.relu(h)
        h = self.enc2(h)

        # decoder
        h = self.dec1(h)
        h = self.relu(h)
        h = self.dec2(h)
        out = self.tanh(h)

        return self.denormalize(out)

    def loss(self, adv_x: th.Tensor, x: th.Tensor, beta: float = 1.0):
        all_params = th.cat([param.view(-1) for param in self.parameters()])
        l1_reg = th.mean(th.abs(all_params))

        expected_out = x
        out = self.forward(adv_x)

        recon_loss = th.mean(th.abs(expected_out - out))
        l1_loss = beta * 0.01 * l1_reg
        loss = recon_loss + l1_loss

        return {"loss": loss, "recon_loss": recon_loss, "reg_loss": l1_loss}

    def repair(self, obs):
        obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
        pred_obs = self.forward(obs_tensor)

        return pred_obs.detach().cpu().numpy().flatten()


class AtlaPolicy(nn.Module):
    def __init__(self, env_id, algo):
        super(AtlaPolicy, self).__init__()
        config = defense_config["atla"][env_id]
        self.fc1 = nn.Linear(config["obs_dim"], config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.fc3 = nn.Linear(config["hidden_size"], config["action_dim"])
        self.relu = nn.ReLU()

        self.algo = algo
        self.to(default_device)

    def forward(self, x: th.Tensor):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out


class AtlaPPO(AtlaPolicy):
    def __init__(self, env_id, algo):
        super(AtlaPPO, self).__init__(env_id, algo)
        self.var = defense_config["atla"][env_id]["ppo_var"]

    def predict(self, obs: np.ndarray):
        obs = obs.reshape(1, obs.shape[0])
        obs_tensor = th.tensor(obs, dtype=th.float32, device=default_device)
        mu = self(obs_tensor).detach().cpu().numpy()
        noise = self.var * np.random.uniform(-1, 1, size=mu.shape)
        action = mu + noise

        return action, None


class AtlaTD3(AtlaPolicy):
    def predict(self, obs: np.ndarray):
        obs = obs.reshape(1, obs.shape[0])
        obs_tensor = th.tensor(obs, dtype=th.float32, device=default_device)
        action = self(obs_tensor).detach().cpu().numpy()

        return action, None
