##
## IQLearn implementation for ICRA lambda discrepancy paper
##
## based on SAC of CleanRL: https://github.com/vwxyzjn/cleanrl
##
## modified based on the papers
##  - https://arxiv.org/abs/2106.12142 IQLearn
##  - https://arxiv.org/abs/2303.00599 LS-IQ
##  - https://arxiv.org/abs/2407.07333
##
## docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch.distributions.categorical import Categorical
from torch.nn.modules import LSTMCell
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.rich import tqdm

from lambda_imitation.recorder_wrapper import RecorderSample, RecorderWrapper


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "auto"
    """device to be used"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: object = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    tensorboard_dir: str | None = "runs"
    """where to store tensorboard logs, if None, no tensorboard will be used"""

    # Algorithm specific arguments
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    use_targets: bool = True
    """Whether or not to use target nets"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.6
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    auto_target_entropy: bool = False
    """whether or not to choose the target entropy automatically"""
    target_entropy: float = -1.0
    """The target entropy when not chosen automatically"""
    hidden_state_recalculation_interval: int = 500
    """How often the hidden states of the demonstration buffer are recalculated"""
    recalculate_hidden_states_in_update: bool = False
    """Whether or not to recalculate hidden states at every step for sample"""
    use_lambda_discrepancy: bool = False
    """Whether or not to also approximate the value function via MC estimation and use lambda discrepancy to optimize memory"""


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_state_dim):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.env = env
        if type(self.env.action_space) == gym.spaces.Box:
            if hidden_state_dim > 0:
                assert (
                    hidden_state_dim % 2 == 0
                ), "hidden_state dimensions have to be even for LSTM"
                self.lstm = LSTMCell(
                    env.observation_space.shape[0] + env.action_space.shape[0],
                    hidden_state_dim // 2,
                )
                self.fc2 = nn.Linear(hidden_state_dim // 2, 32)

            else:
                self.fc1 = nn.Linear(
                    env.observation_space.shape[0] + env.action_space.shape[0], 32
                )
                self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc4 = nn.Linear(32, 1)
        elif type(self.env.action_space) == gym.spaces.Discrete:
            if hidden_state_dim > 0:
                assert (
                    hidden_state_dim % 2 == 0
                ), "hidden_state dimensions have to be even for LSTM"
                self.lstm = LSTMCell(
                    env.observation_space.shape[0],
                    hidden_state_dim // 2,
                )
                self.fc2 = nn.Linear(hidden_state_dim // 2, 32)

            else:
                self.fc1 = nn.Linear(env.observation_space.shape[0], 32)
                self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc4 = nn.Linear(32, self.env.action_space.n)

    def forward(self, x, a, h, gather=True):
        hidden_state = None
        if type(self.env.action_space) == gym.spaces.Box:
            x = torch.cat([x, a], 1)
            if self.hidden_state_dim > 0:
                ht, ct = self.lstm(
                    x,
                    (
                        h[:, : self.hidden_state_dim // 2],
                        h[:, self.hidden_state_dim // 2 :],
                    ),
                )
                x = ht
                hidden_state = torch.cat((ht, ct), dim=-1)
            else:
                x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x, hidden_state
        else:
            if self.hidden_state_dim > 0:
                ht, ct = self.lstm(
                    x,
                    (
                        h[:, : self.hidden_state_dim // 2],
                        h[:, self.hidden_state_dim // 2 :],
                    ),
                )
                x = ht
                hidden_state = torch.cat((ht, ct), dim=-1)
            else:
                x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            if gather:
                return x.gather(1, a.unsqueeze(1).long()).view(-1), hidden_state
            else:
                return x, hidden_state


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, iqlearn, net=None, hidden_state_dim=0):
        super().__init__()
        self.net = None
        self.iqlearn = iqlearn
        self.env = env
        self.hidden_state_dim = hidden_state_dim
        if net is None:
            if type(self.env.action_space) == gym.spaces.Box:
                if hidden_state_dim > 0:
                    assert (
                        hidden_state_dim % 2 == 0
                    ), "hidden_state dimensions have to be even for LSTM"
                    self.lstm = LSTMCell(
                        env.observation_space.shape[0],
                        hidden_state_dim // 2,
                    )
                    self.fc2 = nn.Linear(hidden_state_dim // 2, 32)

                else:
                    self.fc1 = nn.Linear(env.observation_space.shape[0], 32)
                    self.fc2 = nn.Linear(32, 32)
                self.fc3 = nn.Linear(32, 32)
                self.fc_mean = nn.Linear(32, np.prod(env.action_space.shape))
                self.fc_logstd = nn.Linear(32, np.prod(env.action_space.shape))
            elif type(self.env.action_space) == gym.spaces.Discrete:
                if hidden_state_dim > 0:
                    assert (
                        hidden_state_dim % 2 == 0
                    ), "hidden_state dimensions have to be even for LSTM"
                    self.lstm = LSTMCell(
                        env.observation_space.shape[0],
                        hidden_state_dim // 2,
                    )
                    self.fc2 = nn.Linear(hidden_state_dim // 2, 32)

                else:
                    self.fc1 = nn.Linear(env.observation_space.shape[0], 32)
                    self.fc2 = nn.Linear(32, 32)
                self.fc3 = nn.Linear(32, 32)
                self.fc4 = nn.Linear(32, env.action_space.n)
        else:
            self.net = net

        if type(self.env.action_space) == gym.spaces.Box:
            self.register_buffer(
                "action_scale",
                torch.tensor(
                    (env.action_space.high - env.action_space.low) / 2.0,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "action_bias",
                torch.tensor(
                    (env.action_space.high + env.action_space.low) / 2.0,
                    dtype=torch.float32,
                ),
            )

    def forward(self, x, h):
        hidden_state = None
        if self.net is None:
            if self.hidden_state_dim > 0:
                ht, ct = self.lstm(
                    x,
                    (
                        h[..., : self.hidden_state_dim // 2],
                        h[..., self.hidden_state_dim // 2 :],
                    ),
                )
                x = ht
                hidden_state = torch.cat((ht, ct), dim=-1)
            else:
                x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            if type(self.env.action_space) == gym.spaces.Box:
                mean = self.fc_mean(x)
                log_std = self.fc_logstd(x)
                log_std = torch.tanh(log_std)
                log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                    log_std + 1
                )  # From SpinUp / Denis Yarats
            else:
                logits = self.fc4(x)
                return logits, hidden_state
        else:
            mean, log_std = self.net(x)

        return mean, log_std, hidden_state

    def get_action(self, x, h):
        if type(self.env.action_space) == gym.spaces.Box:
            mean, log_std, hidden_state = self(x, h)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias

            return action, log_prob, mean, hidden_state
        else:
            logits, hidden_state = self(x, h)
            policy_dist = Categorical(logits=logits)
            actions = policy_dist.sample()
            action_probs = policy_dist.probs
            log_prob = F.log_softmax(logits, dim=1)
            entropy = torch.sum(action_probs * log_prob, dim=-1)
            greedy_actions = torch.argmax(action_probs, dim=-1)

            return actions, entropy, greedy_actions, hidden_state

    def get_action_probs(self, x, a, h):
        if type(self.env.action_space) == gym.spaces.Box:
            mean, log_std, _ = self(x, h)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = torch.atanh((a - self.action_bias) / self.action_scale)

            return normal.prob(x_t)
        else:
            logits, _ = self(x, h)
            policy_dist = Categorical(logits=logits)
            action_probs = policy_dist.probs

            return action_probs.gather(1, a.unsqueeze(1).long()).view(-1)

    def get_prob_log(self, x, h):
        assert isinstance(
            self.env.action_space, gym.spaces.Discrete
        ), "probabilities and log prob only supported for discrete action spaces!"

        logits, _ = self(x, h)
        policy_dist = Categorical(logits=logits)
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action_probs, log_prob

    def get_actor_loss(self, observations, hidden_state):
        if type(self.env.action_space) == gym.spaces.Box:
            pi, log_pi, _, _ = self.get_action(
                observations, self.iqlearn.get_actor_hidden_state(hidden_state)
            )
            qf1_pi, _ = self.iqlearn.qf1(
                observations, pi, self.iqlearn.get_qf1_hidden_state(hidden_state)
            )
            qf2_pi, _ = self.iqlearn.qf2(
                observations, pi, self.iqlearn.get_qf2_hidden_state(hidden_state)
            )
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            return ((self.iqlearn.alpha * log_pi) - min_qf_pi).mean()
        else:
            qf1_pi, _ = self.iqlearn.qf1(
                observations,
                None,
                self.iqlearn.get_qf1_hidden_state(hidden_state),
                False,
            )
            qf2_pi, _ = self.iqlearn.qf2(
                observations,
                None,
                self.iqlearn.get_qf2_hidden_state(hidden_state),
                False,
            )
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            action_probs, log_prob = self.get_prob_log(
                observations, self.iqlearn.get_actor_hidden_state(hidden_state)
            )
            return (
                (action_probs * ((self.iqlearn.alpha * log_prob) - min_qf_pi)).sum(-1)
            ).mean()


def default_phi(x):
    return x


def default_regularizer(x):
    return x**2 / 10


class IQLearn:
    def __init__(
        self,
        env: gym.Env,
        phi: Callable[[torch.Tensor], torch.Tensor] | None = None,
        regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        online_size: int = 256,
        actor_net: nn.Module | None = None,
        q_cls: type = SoftQNetwork,
        sac_args: Args | dict[str, Any] | None = None,
        hidden_state_dims=(0, 0, 0),
        hidden_state_net=None,
    ):
        if sac_args is None:
            self.args = Args()
        elif type(sac_args) == dict:
            self.args = Args(**sac_args)
        elif type(sac_args) == Args:
            self.args = sac_args

        self.online_size = online_size
        self.demonstration_buffer = None

        self.hidden_state_dims = hidden_state_dims
        self.shared_hidden_states = len(self.hidden_state_dims) == 1
        if self.shared_hidden_states:
            self.actor_hidden_state_dim = self.hidden_state_dims[0]
            self.qf1_hidden_state_dim = self.hidden_state_dims[0]
            self.qf2_hidden_state_dim = self.hidden_state_dims[0]
            self.get_actor_hidden_state = lambda x: x[0]
            self.get_qf1_hidden_state = lambda x: x[0]
            self.get_qf2_hidden_state = lambda x: x[0]
        else:
            self.actor_hidden_state_dim = self.hidden_state_dims[0]
            self.qf1_hidden_state_dim = self.hidden_state_dims[1]
            self.qf2_hidden_state_dim = self.hidden_state_dims[2]
            self.get_actor_hidden_state = lambda x: x[0]
            self.get_qf1_hidden_state = lambda x: x[1]
            self.get_qf2_hidden_state = lambda x: x[2]

        if phi is None:
            self.phi = default_phi
        else:
            self.phi = phi
        if regularizer is None:
            self.regularizer = default_regularizer
        else:
            self.regularizer = regularizer

        if self.args.track:
            import wandb

            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        assert isinstance(env.action_space, gym.spaces.Box) or isinstance(
            env.action_space, gym.spaces.Discrete
        ), "only discrete or continuous action space is supported"
        assert isinstance(
            env.observation_space, gym.spaces.Box
        ), "only continuous observation space is supported"

        self.actor = Actor(
            env, self, actor_net, hidden_state_dim=self.actor_hidden_state_dim
        ).to(self.args.device)
        self.qf1 = q_cls(env, hidden_state_dim=self.qf1_hidden_state_dim).to(
            self.args.device
        )
        self.qf2 = q_cls(env, hidden_state_dim=self.qf2_hidden_state_dim).to(
            self.args.device
        )
        self.mc_qf1 = None
        self.mc_qf2 = None
        if self.args.use_lambda_discrepancy:
            self.mc_qf1 = q_cls(env, hidden_state_dim=self.qf1_hidden_state_dim).to(
                self.args.device
            )
            self.mc_qf2 = q_cls(env, hidden_state_dim=self.qf2_hidden_state_dim).to(
                self.args.device
            )
        if self.shared_hidden_states:
            self.qf1.lstm = self.actor.lstm
            self.qf2.lstm = self.actor.lstm
            if self.args.use_lambda_discrepancy:
                self.mc_qf1.lstm = self.actor.lstm
                self.mc_qf2.lstm = self.actor.lstm
        if self.args.use_targets:
            self.qf1_target = q_cls(env, hidden_state_dim=self.qf1_hidden_state_dim).to(
                self.args.device
            )
            self.qf2_target = q_cls(env, hidden_state_dim=self.qf2_hidden_state_dim).to(
                self.args.device
            )
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
        else:
            self.qf1_target = self.qf1
            self.qf2_target = self.qf2

        if hidden_state_net is None:

            def hidden_state_net(x, a, h):
                if isinstance(x, torch.Tensor):
                    x = x.detach().clone()
                    a = a.detach().clone()
                    h = tuple(s.detach().clone() for s in h)
                else:
                    x = torch.tensor(x).to(self.args.device)
                    a = torch.tensor(a).to(self.args.device)
                    h = tuple(torch.tensor(s).to(self.args.device) for s in h)

                def get_hidden_state(lstm, input, hidden_state):
                    h, c = lstm(
                        input,
                        (
                            hidden_state[..., : hidden_state.shape[-1] // 2],
                            hidden_state[..., hidden_state.shape[-1] // 2 :],
                        ),
                    )
                    return torch.cat((h, c), dim=-1).detach().cpu().numpy()

                xa = x
                if self.qf1_hidden_state_dim > 0 or self.qf2_hidden_state_dim > 0:
                    if type(self.env.action_space) == gym.spaces.Box:
                        xa = torch.cat([x, a], -1)

                if not self.shared_hidden_states:
                    return (
                        (
                            get_hidden_state(
                                self.actor.lstm, x, self.get_actor_hidden_state(h)
                            )
                            if self.hidden_state_dims[0] > 0
                            else np.array((0,))
                        ),
                        (
                            get_hidden_state(
                                self.qf1_target.lstm, xa, self.get_qf1_hidden_state(h)
                            )
                            if self.hidden_state_dims[1] > 0
                            else np.array((0,))
                        ),
                        (
                            get_hidden_state(
                                self.qf2_target.lstm, xa, self.get_qf2_hidden_state(h)
                            )
                            if self.hidden_state_dims[2] > 0
                            else np.array((0,))
                        ),
                    )
                else:
                    return (
                        (
                            get_hidden_state(
                                self.actor.lstm, x, self.get_actor_hidden_state(h)
                            )
                            if self.hidden_state_dims[0] > 0
                            else np.array((0,))
                        ),
                    )

            self.hidden_state_net = hidden_state_net
        else:
            self.hidden_state_net = hidden_state_net

        # Automatic entropy tuning
        if self.args.autotune:
            if self.args.auto_target_entropy:
                self.target_entropy = -torch.prod(
                    torch.Tensor(env.action_space.shape).to(self.args.device)
                ).item()
            else:
                self.target_entropy = self.args.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha
            self.a_optimizer = None
        self.recreate_optimizers()

        self.set_env(env)
        self.setup_writer()

        self.env.observation_space.dtype = np.float32  # type: ignore
        self.start_time = time.time()

        self.n_updates = 0

    def recreate_optimizers(self):
        q_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        if self.args.use_lambda_discrepancy:
            q_parameters += list(self.mc_qf1.parameters()) + list(
                self.mc_qf2.parameters()
            )
            self.lambda_optimizer = optim.Adam(self.actor.lstm.parameters(), lr=self.args.policy_lr)

        self.q_optimizer = optim.Adam(q_parameters, lr=self.args.q_lr)
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.args.policy_lr
        )
        if self.args.autotune:
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)

    def reset_replay_buffer(self):
        self.env = RecorderWrapper(
            self.env.env,
            self.args.gamma,
            self.args.buffer_size,
            self.env.hidden_state_dims,
            self.env.hidden_state_net,
        )

    def setup_writer(self):
        self.run_name = f"{self.env.spec.id if self.env is not None and self.env.spec is not None else ''}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        if self.args.tensorboard_dir is None:
            self.writer = None
        else:
            self.writer = SummaryWriter(f"{self.args.tensorboard_dir}/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % (
                    "\n".join(
                        [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                    )
                ),
            )

    def set_env(self, env):
        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = RecorderWrapper(
            self.env,
            self.args.gamma,
            self.args.buffer_size,
            self.hidden_state_dims,
            self.hidden_state_net,
        )
        self.setup_writer()

    def set_demonstration_buffer(self, demonstration_buffer):
        self.demonstration_buffer = demonstration_buffer
        self.demonstration_buffer.hidden_state_net = self.hidden_state_net
        assert (
            self.demonstration_buffer.hidden_state_dims == self.hidden_state_dims
        ), "demonstration buffer has feature same hidden state dims as training"

    def learn(self, timesteps: int, progress="tqdm"):
        # ALGO LOGIC: put action logic here
        if self.online_size > 0:
            obs, _ = self.env.reset(seed=np.random.randint(2147483647))
            hidden_state = torch.zeros(
                (1, self.actor_hidden_state_dim), dtype=torch.float32
            ).to(self.args.device)

        if progress == "tqdm":
            it = tqdm(range(timesteps))
        else:
            it = range(timesteps)
        for _ in it:
            if self.online_size > 0:
                if self.n_updates < self.args.learning_starts:
                    action = np.array(self.env.action_space.sample())
                else:
                    action, _, _, hidden_state = self.actor.get_action(
                        torch.Tensor(obs).unsqueeze(0).to(self.args.device),
                        hidden_state,
                    )
                    action = action.detach().cpu().numpy()[0]

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, termination, truncated, info = self.env.step(action)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in info and self.writer is not None:
                    for info in info["final_info"]:
                        # print(
                        #     f"self.n_updates={self.n_updates}, episodic_return={info['episode']['r']}"
                        # )
                        self.writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            self.n_updates,
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            self.n_updates,
                        )
                        break

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
                if termination or truncated:
                    obs, _ = self.env.reset(seed=np.random.randint(2147483647))
                    hidden_state = torch.zeros(
                        (1, self.actor_hidden_state_dim), dtype=torch.float32
                    ).to(self.args.device)

            # ALGO LOGIC: training.
            if self.n_updates > self.args.learning_starts:
                if self.n_updates % self.args.hidden_state_recalculation_interval == 0:
                    self.demonstration_buffer.recalculate_hidden_states()
                data = self.demonstration_buffer.sample(
                    self.args.batch_size,
                    self.args.device,
                    full_episodes_only=self.args.use_lambda_discrepancy,
                )
                if self.args.recalculate_hidden_states_in_update:
                    next_hidden_states = self.hidden_state_net(
                        data.observations, data.actions, data.hidden_states
                    )
                    next_hidden_states = (
                        tuple(  # unelegant, but hidden_state_net returns numpy array
                            torch.tensor(hidden_state).to(self.args.device)
                            for hidden_state in next_hidden_states
                        )
                    )
                    data = RecorderSample(
                        data.observations,
                        data.next_observations,
                        data.actions,
                        data.rewards,
                        data.returns,
                        data.terminated,
                        data.truncated,
                        data.hidden_states,
                        next_hidden_states,
                    )
                    self.demonstration_buffer.override_next_hidden_states_last_sample(
                        data.next_hidden_states
                    )

                loss, demonstration_loss, mixed_loss, regularizer_loss = (
                    self.update_critic(data)
                )
                if (
                    self.n_updates % self.args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    actor_loss, alpha_loss = self.update_policy(data)

                # update the target networks
                if (
                    self.n_updates % self.args.target_network_frequency == 0
                    and self.args.use_targets
                ):
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if self.n_updates % 100 == 0 and self.writer is not None:
                    self.writer.add_scalar(
                        "losses/critic_loss",
                        loss.item(),
                        self.n_updates,
                    )
                    self.writer.add_scalar(
                        "losses/demonstration_loss",
                        demonstration_loss.item(),
                        self.n_updates,
                    )
                    self.writer.add_scalar(
                        "losses/mixed_loss", mixed_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/regularizer_loss",
                        regularizer_loss.item(),
                        self.n_updates,
                    )
                    self.writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), self.n_updates  # type: ignore
                    )
                    self.writer.add_scalar("losses/alpha", self.alpha, self.n_updates)
                    # print(
                    #     "SPS:", int(self.n_updates / (time.time() - self.start_time))
                    # )
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(self.n_updates / (time.time() - self.start_time)),
                        self.n_updates,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), self.n_updates  # type: ignore
                        )
            self.n_updates += 1

    def get_values(self, observations, hidden_states, actions=None):
        if actions is None:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                qf1_pi, _ = self.qf1(
                    observations, None, self.get_qf1_hidden_state(hidden_states), False
                )
                qf2_pi, _ = self.qf2(
                    observations, None, self.get_qf2_hidden_state(hidden_states), False
                )
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                action_probs, _ = self.actor.get_prob_log(
                    observations, hidden_states[0]
                )
                return (action_probs * min_qf_pi).sum(-1)

            actions, _, _, _ = self.actor.get_action(
                observations, self.get_actor_hidden_state(hidden_states)
            )

        qf1_a_values = self.qf1(
            observations, actions, self.get_qf1_hidden_state(hidden_states)
        )[0].view(-1)
        qf2_a_values = self.qf2(
            observations, actions, self.get_qf2_hidden_state(hidden_states)
        )[0].view(-1)
        return torch.min(qf1_a_values, qf2_a_values).unsqueeze(1)

    def update_critic(self, data, live_data=None):
        demonstration_loss = (
            self.get_values(data.observations, data.hidden_states, data.actions)
            - (1 - data.terminated.float())
            * self.args.gamma
            * self.get_values(data.next_observations, data.next_hidden_states).detach()
        )

        mixed_loss = (
            self.get_values(data.observations, data.hidden_states)
            - (1 - data.terminated.float())
            * self.args.gamma
            * self.get_values(data.next_observations, data.next_hidden_states).detach()
        )
        if live_data is not None:
            live_loss = (
                self.get_values(live_data.observations, live_data.hidden_states)
                - (1 - live_data.terminated.float())
                * self.args.gamma
                * self.get_values(
                    live_data.next_observations, live_data.next_hidden_states
                ).detach()
            )
        else:
            live_loss = []  # hack so live_loss has len()

        data_normalizer = (len(mixed_loss) + len(live_loss)) / (len(demonstration_loss))

        regularizer_loss = self.regularizer(mixed_loss).mean()
        if live_data is not None:
            regularizer_loss += self.regularizer(live_loss).mean()  # type: ignore
        regularizer_loss += self.regularizer(demonstration_loss).mean()

        demonstration_loss = self.phi(demonstration_loss).mean()
        mixed_loss = (
            mixed_loss.mean() + (0 if live_data is None else live_loss.mean())  # type: ignore
        ) / data_normalizer
        regularizer_loss = regularizer_loss.mean()

        loss = demonstration_loss - mixed_loss - regularizer_loss
        loss = -loss  # maximize

        # optimize the model
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss, demonstration_loss, mixed_loss, regularizer_loss

    def update_policy(self, data):
        actor_loss = 0
        alpha_loss = 0
        for _ in range(
            self.args.policy_frequency
        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
            actor_loss = self.actor.get_actor_loss(
                data.observations, data.hidden_states
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.args.autotune:
                with torch.no_grad():
                    _, log_pi, _, _ = self.actor.get_action(
                        data.observations,
                        self.get_actor_hidden_state(data.hidden_states),
                    )
                    probs = self.actor.get_action_prob(
                        data.observations,
                        data.actions,
                        self.get_actor_hidden_state(data.hidden_states),
                    )
                    self.env.override_policy_probabilities_last_sample(
                        probs.detach().cpu().numpy()
                    )
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi + self.target_entropy)
                ).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        return actor_loss, alpha_loss

    def predict(
        self,
        obs: torch.Tensor | np.ndarray,
        hidden_state=None,
        deterministic: bool = False,
    ):
        if type(obs) == np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.args.device)
        if type(hidden_state) == np.ndarray:
            hidden_state = torch.tensor(
                hidden_state, dtype=torch.float32, device=self.args.device
            )
        obs = obs.unsqueeze(0)  # type: ignore
        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(0)  # type: ignore
        action, _, mean, hidden_state = self.actor.get_action(obs, hidden_state)
        prediction = mean if deterministic else action
        prediction = prediction.detach().cpu().numpy()
        prediction = prediction[0]
        if hidden_state is not None:
            hidden_state = hidden_state.detach().cpu().numpy()
            hidden_state = self.get_actor_hidden_state(hidden_state)
        return prediction, hidden_state

    def sac_learn(self, steps, progress="tqdm"):
        obs, _ = self.env.reset()
        hidden_state = torch.zeros(
            (1, self.actor_hidden_state_dim), dtype=torch.float32
        ).to(self.args.device)
        if progress == "tqdm":
            it = tqdm(range(steps))
        else:
            it = range(steps)
        for _ in it:
            # ALGO LOGIC: put action logic here
            # if self.n_updates < self.args.learning_starts:
            #     action = np.array(self.env.action_space.sample())
            # else:
            torch_obs = torch.Tensor(obs).unsqueeze(0).to(self.args.device)
            action, _, _, hidden_state = self.actor.get_action(
                torch_obs,
                hidden_state,
            )
            probs = self.actor.get_action_probs(torch_obs, action, hidden_state)
            action = action.detach().cpu().numpy()[0]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncated, info = self.env.step(action)
            self.env.set_probabilities_of_last_action(probs[0])
            self.env.recalculate_episodes()

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if (termination or truncated) and self.writer is not None:
                if info is not None:
                    self.writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], self.n_updates
                    )
                    self.writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], self.n_updates
                    )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if termination or truncated:
                obs, _ = self.env.reset()

            # ALGO LOGIC: training.
            if self.n_updates > self.args.learning_starts:
                data = self.env.sample(
                    self.args.batch_size,
                    self.args.device,
                    full_episodes_only=self.args.use_lambda_discrepancy,
                )
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _, _ = self.actor.get_action(
                        data.next_observations,
                        self.get_actor_hidden_state(data.hidden_states),
                    )
                    qf1_next_target, _ = self.qf1_target(
                        data.next_observations,
                        next_state_actions,
                        self.get_qf1_hidden_state(data.hidden_states),
                    )
                    qf2_next_target, _ = self.qf2_target(
                        data.next_observations,
                        next_state_actions,
                        self.get_qf2_hidden_state(data.hidden_states),
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.terminated.float().flatten()
                    ) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values, _ = self.qf1(
                    data.observations,
                    data.actions,
                    self.get_qf1_hidden_state(data.hidden_states),
                )
                qf1_a_values = qf1_a_values.view(-1)
                qf2_a_values, _ = self.qf2(
                    data.observations,
                    data.actions,
                    self.get_qf2_hidden_state(data.hidden_states),
                )
                qf2_a_values = qf2_a_values.view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                if self.args.use_lambda_discrepancy:
                    mc_qf1_a_values, _ = self.mc_qf1(
                        data.observations,
                        data.actions,
                        self.get_qf1_hidden_state(data.hidden_states),
                    )
                    mc_qf1_a_values = mc_qf1_a_values.view(-1)
                    mc_qf2_a_values, _ = self.mc_qf2(
                        data.observations,
                        data.actions,
                        self.get_qf2_hidden_state(data.hidden_states),
                    )
                    mc_qf2_a_values = mc_qf2_a_values.view(-1)

                    lambda_discrepancy = (
                        torch.min(qf1_a_values, qf2_a_values)
                        - torch.min(mc_qf1_a_values, mc_qf2_a_values)
                    ) ** 2
                    lambda_discrepancy = lambda_discrepancy.mean()

                    print(data.importance_factors.mean())
                    mc_qf1_loss = F.mse_loss(
                        mc_qf1_a_values, data.returns, reduction='none'
                    )
                    mc_qf2_loss = F.mse_loss(
                        mc_qf2_a_values, data.returns, reduction='none'
                    )
                    qf_loss += (data.importance_factors*(mc_qf1_loss + mc_qf2_loss)).mean()

                    if self.n_updates % 100 == 0 and self.writer is not None:
                        self.writer.add_scalar(
                            "charts/returns",
                            data.returns.mean().detach().cpu().numpy().item(),
                            self.n_updates,
                        )
                        self.writer.add_scalar(
                            "losses/mc_qf1_values",
                            mc_qf1_a_values.mean().item(),
                            self.n_updates,
                        )
                        self.writer.add_scalar(
                            "losses/mc_qf2_values",
                            mc_qf2_a_values.mean().item(),
                            self.n_updates,
                        )
                        self.writer.add_scalar(
                            "losses/mc_qf1_loss", mc_qf1_loss.mean().item(), self.n_updates
                        )
                        self.writer.add_scalar(
                            "losses/mc_qf2_loss", mc_qf2_loss.mean().item(), self.n_updates
                        )
                        self.writer.add_scalar(
                            "charts/lambda-discrepancy",
                            lambda_discrepancy.item(),
                            self.n_updates,
                        )

                if (
                    self.n_updates % self.args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        self.args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1

                        actor_loss = self.actor.get_actor_loss(
                            data.observations, data.hidden_states
                        )

                        qf_loss += actor_loss
                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _, _ = self.actor.get_action(
                                    data.observations,
                                    self.get_actor_hidden_state(data.hidden_states),
                                )
                                probs = self.actor.get_action_probs(
                                    data.observations,
                                    data.actions,
                                    self.get_actor_hidden_state(data.hidden_states),
                                )
                                self.env.override_policy_probabilities_last_sample(
                                    probs.detach().cpu().numpy()
                                )
                            alpha_loss = (
                                -self.log_alpha.exp() * (log_pi + self.target_entropy)
                            ).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # optimize the model
                self.q_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                if self.args.use_lambda_discrepancy:
                    self.lambda_optimizer.zero_grad()
                    lambda_discrepancy.backward(retain_graph=True)
                qf_loss.backward()
                if self.args.use_lambda_discrepancy:
                    self.lambda_optimizer.step()
                self.actor_optimizer.step()
                self.q_optimizer.step()

                # update the target networks
                if self.n_updates % self.args.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if self.n_updates % 100 == 0 and self.writer is not None:
                    self.writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar("losses/alpha", self.alpha, self.n_updates)
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(self.n_updates / (time.time() - self.start_time)),
                        self.n_updates,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), self.n_updates
                        )
                        self.writer.add_scalar(
                            "losses/entropy", (-log_pi).mean().item(), self.n_updates
                        )
            self.n_updates += 1

    def close(self):
        self.env.close()
        if self.writer is not None:
            self.writer.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["writer"]
        del state["env"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.writer = None
        self.env = None
