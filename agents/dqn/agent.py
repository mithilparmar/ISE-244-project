import copy
from typing import Callable
import numpy as np
import torch
from torch import nn

# pylint: disable=import-error
import agents.replay as replay_lib
import agents.types as types_lib
import agents.value_learning as rl
from agents import base

# torch.autograd.set_detect_anomaly(True)


class Dqn(types_lib.Agent):
    """DQN agent"""

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        replay: replay_lib.UniformReplay,
        transition_accumulator: replay_lib.NStepTransitionAccumulator,
        exploration_epsilon: Callable[[int], float],
        learn_frequency: int,
        target_network_update_frequency: int,
        min_replay_size: int,
        batch_size: int,
        num_actions: int,
        n_step: int,
        discount: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ):
        """
        Args:
            network: the Q network we want to optimize.
            optimizer: the optimizer for Q network.
            random_state: used to sample random actions for e-greedy policy.
            replay: experience replay.
            transition_accumulator: external helper class to build n-step transition.
            exploration_epsilon: external schedul of e in e-greedy exploration rate.
            learn_frequency: the frequency (measured in agent steps) to do learning.
            target_network_update_frequency: the frequency (measured in number of online Q network parameter updates)
                 to update target Q network weights.
            min_replay_size: Minimum replay size before start to do learning.
            batch_size: sample batch size.
            num_actions: number of valid actions in the environment.
            n_step: TD n-step bootstrap.
            discount: gamma discount for future rewards.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 1 <= learn_frequency:
            raise ValueError(f'Expect learn_frequency to be positive integer, got {learn_frequency}')
        if not 1 <= target_network_update_frequency:
            raise ValueError(
                f'Expect target_network_update_frequency to be positive integer, got {target_network_update_frequency}'
            )
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be positive integer, got {min_replay_size}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size [1, 512], got {batch_size}')
        if not batch_size <= min_replay_size <= replay.capacity:
            raise ValueError(f'Expect min_replay_size >= {batch_size} and <= {replay.capacity} and, got {min_replay_size}')
        if not 0 < num_actions:
            raise ValueError(f'Expect num_actions to be positive integer, got {num_actions}')
        if not 1 <= n_step:
            raise ValueError(f'Expect n_step to be positive integer, got {n_step}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount [0.0, 1.0], got {discount}')

        self.agent_name = 'DQN'

        self._device = device
        self._random_state = random_state
        self._num_actions = num_actions

        # Online Q network
        self._online_network = network.to(device=self._device)
        self._optimizer = optimizer

        # Lazy way to create target Q network
        self._target_network = copy.deepcopy(self._online_network).to(device=self._device)
        # Disable autograd for target network
        for p in self._target_network.parameters():
            p.requires_grad = False

        # Experience replay parameters
        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._replay = replay

        # Learning related parameters
        self._discount = discount
        self._n_step = n_step
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_size = min_replay_size
        self._learn_frequency = learn_frequency
        self._target_network_update_frequency = target_network_update_frequency
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters
        self._step_t = -1
        self._update_t = -1
        self._target_update_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, do a action selection and a series of learn related activities"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and add into replay
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._replay.add(transition)

        # Return if replay is not ready
        if self._replay.size < self._min_replay_size:
            return a_t

        # Start to learn
        if self._step_t % self._learn_frequency == 0:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        a_t = self._choose_action(timestep, self.exploration_epsilon)
        return a_t

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep, epsilon: float) -> types_lib.Action:
        """
        Choose action by following the e-greedy policy with respect to Q values

        Args:
            timestep: the current timestep from env
            epsilon: the e in e-greedy exploration
        Returns:
            a_t: the action to take at s_t
        """

        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._num_actions)
            return a_t

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_values = self._online_network(s_t).q_values
        a_t = torch.argmax(q_values, dim=-1)
        return a_t.cpu().item()

    def _learn(self) -> None:
        """Sample a batch of transitions and learn."""
        transitions = self._replay.sample(self._batch_size)
        self._update(transitions)

        # Update target Q network weights
        if self._update_t % self._target_network_update_frequency == 0:
            self._update_target_network()

    def _update(self, transitions: replay_lib.Transition) -> None:
        self._optimizer.zero_grad()
        loss = self._calc_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._online_network.parameters(), self._max_grad_norm, error_if_nonfinite=True)
        self._optimizer.step()
        self._update_t += 1

    def _calc_loss(self, transitions: replay_lib.Transition) -> torch.Tensor:
        """Calculate loss for a given batch of transitions."""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 1, torch.long)
        base.assert_rank_and_dtype(r_t, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)

        discount_t = (~done).float() * self._discount**self._n_step

        # Compute predicted q values for s_tm1, using online Q network
        q_tm1 = self._online_network(s_tm1).q_values  # [batch_size, num_actions]

        # Compute predicted q values for s_t, using target Q network
        with torch.no_grad():
            target_q_t = self._target_network(s_t).q_values  # [batch_size, num_actions]

        # Compute loss which is 0.5 * square(td_errors)
        loss = rl.qlearning(q_tm1, a_tm1, r_t, discount_t, target_q_t).loss
        # Average over batch dimension
        loss = torch.mean(loss, dim=0)
        return loss

    def _update_target_network(self):
        """Copy online network weights to target network."""
        self._target_network.load_state_dict(self._online_network.state_dict())
        self._target_update_t += 1

    @property
    def exploration_epsilon(self):
        """Call external schedule function"""
        return self._exploration_epsilon(self._step_t)

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'discount': self._discount,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
            'exploration_epsilon': self.exploration_epsilon,
        }
