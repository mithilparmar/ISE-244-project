from typing import Mapping, Optional, Tuple, NamedTuple
import queue
import copy
import multiprocessing
import numpy as np
import torch
from torch import nn


# pylint: disable=import-error
import agents.replay as replay_lib
import agents.types as types_lib
from agents import base
from agents import multistep
from agents import distributed
from agents import transforms
from agents.networks.dqn import RnnDqnNetworkInputs

# torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class R2d2Transition(NamedTuple):
    r"""Ideally we want to construct transition in the manner of (s, a, r), this is not the case for gym env.
    when the agent takes action 'a' in state 's', it does not receive rewards immediately, only at the next time step.
    so there's going to be one timestep lag between the (s, a, r).

    For simplicity reasons, we choose to lag rewards, so (s, a, r) is actually (s', a', r).

    In our case 'a_t', 'done' is for 's_t', while
    'r_t' is for previous state-action (s_tm1, a_tm1) but received at current timestep.

    Note 'a_tm1' is only used for network pass during learning,
    'q_t' is only for calculating priority for the unroll sequence when adding into replay."""

    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t
    r_t: Optional[float]
    a_tm1: Optional[int]
    done: Optional[bool]
    init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state
    init_c: Optional[np.ndarray]  # nn.LSTM initial cell state


TransitionStructure = R2d2Transition(
    s_t=None,
    a_t=None,
    q_t=None,
    r_t=None,
    a_tm1=None,
    done=None,
    init_h=None,
    init_c=None,
)


def no_autograd(net: torch.nn.Module):
    """Disable autograd for a network."""
    for p in net.parameters():
        p.requires_grad = False


def calculate_losses_and_priorities(
    q_value: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    target_qvalue: torch.Tensor,
    target_action: torch.Tensor,
    gamma: float,
    n_step: int,
    eps: float = 0.001,
    eta: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate loss and priority for given samples.

    T is the unrolled length, B the batch size, N is number of actions.

    Args:
        q_value: (T+1, B, num_actions) the predicted q values for a given state 's_t' from online Q network.
        action: [T+1, B] the actual action the agent take in state 's_t'.
        reward: [T+1, B] the reward the agent received at timestep t, this is for (s_tm1, a_tm1).
        done: [T+1, B] terminal mask for timestep t, state 's_t'.
        target_qvalue: (T+1, B, N) the estimated TD n-step target values from target Q network,
            this could also be the same q values when just calculate priorities to insert into replay.
        target_action: [T+1, B] the best action to take in t+n timestep target state.
        gamma: discount rate.
        n_step: TD n-step size.
        eps: constant for value function rescaling and inverse value function rescaling.
        eta: constant for calculate mixture priorities.

    Returns:
        losses: the losses for given unrolled samples, shape (B, )
        priorities: the priority for given samples, shape (B, )
    """

    base.assert_rank_and_dtype(q_value, 3, torch.float32)
    base.assert_rank_and_dtype(target_qvalue, 3, torch.float32)
    base.assert_rank_and_dtype(reward, 2, torch.float32)
    base.assert_rank_and_dtype(action, 2, torch.long)
    base.assert_rank_and_dtype(target_action, 2, torch.long)
    base.assert_rank_and_dtype(done, 2, torch.bool)

    q_value = q_value.gather(-1, action[..., None]).squeeze(-1)  # [T, B]

    target_q_max = target_qvalue.gather(-1, target_action[..., None]).squeeze(-1)  # [T, B]
    # Apply invertible value rescaling to TD target.
    target_q_max = transforms.signed_parabolic(target_q_max, eps)

    # Note the input rewards into 'n_step_bellman_target' should be non-discounted, non-summed.
    target_q = multistep.n_step_bellman_target(r_t=reward, done=done, q_t=target_q_max, gamma=gamma, n_steps=n_step)

    # q_value is actually Q(s_t, a_t), but rewards is for 's_tm1', 'a_tm1',
    # that means our 'target_q' value is one step behind 'q_value',
    # so we need to shift them to make it in the same timestep.
    q_value = q_value[:-1, ...]
    target_q = target_q[1:, ...]

    # Apply value rescaling to TD target.
    target_q = transforms.signed_hyperbolic(target_q, eps)

    if q_value.shape != target_q.shape:
        raise RuntimeError(f'Expect q_value and target_q have the same shape, got {q_value.shape} and {target_q.shape}')

    td_error = target_q - q_value

    with torch.no_grad():
        priorities = distributed.calculate_dist_priorities_from_td_error(td_error, eta)

    # Sums over time dimension.
    losses = 0.5 * torch.sum(torch.square(td_error), dim=0)  # [B]

    return losses, priorities


class Actor(types_lib.Agent):
    """R2D2 actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        network: torch.nn.Module,
        learner_network: torch.nn.Module,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        num_actors: int,
        num_actions: int,
        unroll_length: int,
        burn_in: int,
        actor_update_frequency: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank number for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            network: the Q network for actor to make action choice.
            learner_network: the Q network with the updated weights.
            random_state: used to sample random actions for e-greedy policy.
            num_actors: the number actors for calculating e-greedy epsilon.
            num_actions: the number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps.
            actor_update_frequency: the frequency to update actor local Q network.
            device: PyTorch runtime device.
        """
        if not 0 < num_actors:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}')
        if not 0 < num_actions:
            raise ValueError(f'Expect num_actions to be positive integer, got {num_actions}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in to be integer between [0, {unroll_length}), got {burn_in}')
        if not 1 <= actor_update_frequency:
            raise ValueError(
                f'Expect actor_update_frequency to be integer geater than or equal to 1, got {actor_update_frequency}'
            )

        self.rank = rank
        self.agent_name = f'R2D2-actor{rank}'

        self._network = network.to(device=device)
        self._learner_network = learner_network.to(device=device)
        self._update_actor_q_network()

        # Disable autograd for actor's network
        no_autograd(self._network)

        self._queue = data_queue

        self._device = device
        self._random_state = random_state
        self._num_actions = num_actions
        self._actor_update_frequency = actor_update_frequency

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        epsilons = distributed.get_actor_exploration_epsilon(num_actors)
        self._exploration_epsilon = epsilons[self.rank]

        self._a_tm1 = None
        self._lstm_state = None  # Stores nn.LSTM hidden state and cell state

        self._step_t = -1

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        if self._step_t % self._actor_update_frequency == 0:
            self._update_actor_q_network()

        q_t, a_t, hidden_s = self.act(timestep)

        # Note the reward is for s_tm1, a_tm1, because it's only available one agent step after,
        # and the done mark is for current timestep s_t.
        transition = R2d2Transition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            r_t=timestep.reward,
            done=timestep.done,
            a_tm1=self._a_tm1,
            init_h=self._lstm_state[0].squeeze(1).cpu().numpy(),  # remove batch dimension
            init_c=self._lstm_state[1].squeeze(1).cpu().numpy(),
        )
        unrolled_transition = self._unroll.add(transition, timestep.done)
        self._a_tm1, self._lstm_state = a_t, hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()
        self._a_tm1 = self._random_state.randint(0, self._num_actions)  # Initialize a_tm1 randomly
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, Tuple[torch.Tensor]]:
        'Given state s_t and done marks, return an action.'
        return self._choose_action(timestep, self._exploration_epsilon)

    @torch.no_grad()
    def _choose_action(
        self, timestep: types_lib.TimeStep, epsilon: float
    ) -> Tuple[np.ndarray, types_lib.Action, Tuple[torch.Tensor]]:
        """Given state s_t, choose action a_t"""
        pi_output = self._network(self._prepare_network_input(timestep))
        q_t = pi_output.q_values.squeeze()
        a_t = torch.argmax(q_t, dim=-1).cpu().item()

        # To make sure every actors generates the same amount of samples, we apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will geneate less samples.
        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._num_actions)

        return (q_t.cpu().numpy(), a_t, pi_output.hidden_s)

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> RnnDqnNetworkInputs:
        # R2D2 network expect input shape [T, B, state_shape],
        # and additionally 'last action', 'reward for last action', and hidden state from previous timestep.
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._a_tm1).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        return RnnDqnNetworkInputs(
            s_t=s_t[None, ...],  # [T, B, state_shape]
            a_tm1=a_tm1[None, ...],  # [T, B]
            r_t=r_t[None, ...],  # [T, B]
            hidden_s=hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _update_actor_q_network(self):
        self._network.load_state_dict(self._learner_network.state_dict())

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current actor's statistics as a dictionary."""
        return {'exploration_epsilon': self._exploration_epsilon}


class Learner:
    """R2D2 learner"""

    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_network_update_frequency: int,
        min_replay_size: int,
        num_actors: int,
        batch_size: int,
        n_step: int,
        discount: float,
        burn_in: int,
        priority_eta: float,
        rescale_epsilon: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            data_queue: a multiprocessing.Queue to get collected transitions from actor processes.
            network: the Q network we want to train and optimize.
            optimizer: the optimizer for Q network.
            replay: prioritized recurrent experience replay.
            target_network_update_frequency: how often to copy online network weights to target.
            min_replay_size: wait till experience replay buffer this number before start to learn.
            num_actors: number of actor processes.
            batch_size: sample batch_size of transitions.
            n_step: TD n-step bootstrap.
            discount: the gamma discount for future rewards.
            burn_in: burn n transitions to generate initial hidden state before learning.
            priority_eta: coefficient to mix the max and mean absolute TD errors.
            rescale_epsilon: rescaling factor for n-step targets in the invertible rescaling function.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 1 <= target_network_update_frequency:
            raise ValueError(
                f'Expect target_network_update_frequency to be positive integer, got {target_network_update_frequency}'
            )
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer geater than or equal to 1, got {min_replay_size}')
        if not 1 <= num_actors:
            raise ValueError(f'Expect num_actors to be integer geater than or equal to 1, got {num_actors}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= n_step:
            raise ValueError(f'Expect n_step to be integer geater than or equal to 1, got {n_step}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}')
        if not 0.0 <= rescale_epsilon <= 1.0:
            raise ValueError(f'Expect rescale_epsilon to in the range [0.0, 1.0], got {rescale_epsilon}')

        self.agent_name = 'R2D2-learner'
        self._device = device
        self._online_network = network.to(device=device)
        self._optimizer = optimizer
        # Lazy way to create target Q network
        self._target_network = copy.deepcopy(self._online_network).to(device=self._device)
        self._update_target_network()

        # Disable autograd for target network
        no_autograd(self._target_network)

        self._batch_size = batch_size
        self._n_step = n_step
        self._burn_in = burn_in
        self._target_network_update_frequency = target_network_update_frequency
        self._discount = discount
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._rescale_epsilon = rescale_epsilon

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._num_actors = num_actors
        self._queue = data_queue

        # Counters
        self._step_t = -1
        self._update_t = -1
        self._done_actors = 0

    def run_train_loop(
        self,
    ) -> None:
        """Start the learner training loop, only break if all actor processes are done."""
        self.reset()
        while True:
            self._step_t += 1

            # Pull one item off queue
            try:
                item = self._queue.get()
                if item == 'PROCESS_DONE':  # actor process is done
                    self._done_actors += 1
                else:
                    # Use the unrolled sequence to calculate priority
                    priority = self._compute_priority_for_unroll(item)
                    self._replay.add(item, priority)
            except queue.Empty:
                pass
            except EOFError:
                pass

            # Only break if all actor processes are done
            if self._done_actors == self._num_actors:
                break

            if self._replay.size < self._min_replay_size:
                continue

            # Pull a batch of unrolls before next learning
            if self._step_t % self._batch_size == 0:
                self._learn()

    def reset(self) -> None:
        """Should be called at the begining of every iteration."""
        self._done_actors = 0

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update(transitions, weights)

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({self._batch_size},), got {priorities.shape}')
        self._replay.update_priorities(indices, priorities)

        # Copy online Q network weights to target Q network, every m updates
        if self._update_t % self._target_network_update_frequency == 0:
            self._update_target_network()

    def _update(self, transitions: R2d2Transition, weights: np.ndarray) -> np.ndarray:
        """Update online Q network."""
        weights = torch.from_numpy(weights).to(device=self._device, dtype=torch.float32)  # [batch_size]
        base.assert_rank_and_dtype(weights, 1, torch.float32)

        # Get initial hidden state, handle possible burn in.
        init_hidden_state = self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in)
        if burn_transitions is not None:
            hidden_online_q, hidden_target_q = self._burn_in_unroll_q_networks(burn_transitions, init_hidden_state)
        else:
            hidden_online_q = tuple(s.clone().to(device=self._device) for s in init_hidden_state)
            hidden_target_q = tuple(s.clone().to(device=self._device) for s in init_hidden_state)

        self._optimizer.zero_grad()
        # [batch_size]
        loss, priorities = self._calc_loss(learn_transitions, hidden_online_q, hidden_target_q)

        # Multiply loss by sampling weights, averages over batch dimension
        loss = torch.mean(loss * weights.detach())
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._online_network.parameters(), self._max_grad_norm)

        self._optimizer.step()
        self._update_t += 1
        return priorities

    def _calc_loss(
        self,
        transitions: R2d2Transition,
        hidden_online_q: HiddenState,
        hidden_target_q: HiddenState,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Calculate loss and priorities for given unroll sequence transitions."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [T+1, B, state_shape]
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [T+1, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 2, torch.long)
        base.assert_rank_and_dtype(r_t, 2, torch.float32)
        base.assert_rank_and_dtype(done, 2, torch.bool)

        # Get q values from online Q network
        q_t = self._online_network(RnnDqnNetworkInputs(s_t=s_t, a_tm1=a_tm1, r_t=r_t, hidden_s=hidden_online_q)).q_values

        # Computes raw target q values, use double Q
        with torch.no_grad():
            # Get best actions a* for 's_t' from online Q network.
            best_a_t = torch.argmax(q_t, dim=-1)  # [T, B]

            # Get estimated q values for 's_t' from target Q network, using above best action a*.
            target_q_t = self._target_network(
                RnnDqnNetworkInputs(s_t=s_t, a_tm1=a_tm1, r_t=r_t, hidden_s=hidden_target_q)
            ).q_values

        losses, priorities = calculate_losses_and_priorities(
            q_value=q_t,
            action=a_t,
            reward=r_t,
            done=done,
            target_qvalue=target_q_t,
            target_action=best_a_t,
            gamma=self._discount,
            n_step=self._n_step,
            eps=self._rescale_epsilon,
            eta=self._priority_eta,
        )

        return (losses, priorities)

    def _burn_in_unroll_q_networks(
        self, transitions: R2d2Transition, init_hidden_state: HiddenState
    ) -> Tuple[HiddenState, HiddenState]:
        """Unroll both online and target q networks to generate hidden states for LSTM."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [burn_in, B]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 2, torch.long)
        base.assert_rank_and_dtype(r_t, 2, torch.float32)

        hidden_online = tuple(s.clone().to(device=self._device) for s in init_hidden_state)
        hidden_target = tuple(s.clone().to(device=self._device) for s in init_hidden_state)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        with torch.no_grad():
            hidden_online_q = self._online_network(
                RnnDqnNetworkInputs(s_t=s_t, a_tm1=a_tm1, r_t=r_t, hidden_s=hidden_online)
            ).hidden_s
            hidden_target_q = self._target_network(
                RnnDqnNetworkInputs(s_t=s_t, a_tm1=a_tm1, r_t=r_t, hidden_s=hidden_target)
            ).hidden_s

        return (hidden_online_q, hidden_target_q)

    def _extract_first_step_hidden_state(self, transitions: R2d2Transition) -> HiddenState:
        # We only need the first step hidden states in replay, shape [batch_size, num_lstm_layers, lstm_hidden_size]
        init_h = torch.from_numpy(transitions.init_h[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        init_c = torch.from_numpy(transitions.init_c[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)

        # Rank and dtype checks.
        base.assert_rank_and_dtype(init_h, 3, torch.float32)
        base.assert_rank_and_dtype(init_c, 3, torch.float32)

        # Swap batch and num_lstm_layers axis.
        init_h = init_h.swapaxes(0, 1)
        init_c = init_c.swapaxes(0, 1)

        # Batch dimension checks.
        base.assert_batch_dimension(init_h, self._batch_size, 1)
        base.assert_batch_dimension(init_c, self._batch_size, 1)

        return (init_h, init_c)

    @torch.no_grad()
    def _compute_priority_for_unroll(self, transitions: R2d2Transition) -> float:
        """Returns priority for a single unroll, no network pass and gradients are required."""
        # Note we skip the burn in part, and use the same q values for target.
        _, learn_transition = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in)

        a_t = torch.from_numpy(learn_transition.a_t).to(device=self._device, dtype=torch.int64)  # [T+1, ]
        q_t = torch.from_numpy(learn_transition.q_t).to(device=self._device, dtype=torch.float32)  # [T+1, ]
        r_t = torch.from_numpy(learn_transition.r_t).to(device=self._device, dtype=torch.float32)  # [T+1, ]
        done = torch.from_numpy(learn_transition.done).to(device=self._device, dtype=torch.bool)  # [T+1, ]

        # Rank and dtype checks, single unroll should not have batch dimension.
        base.assert_rank_and_dtype(q_t, 2, torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(r_t, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)

        # Calculate loss and priority, needs to add a batch dimension.
        _, prioritiy = calculate_losses_and_priorities(
            q_value=q_t.unsqueeze(1),
            action=a_t.unsqueeze(1),
            reward=r_t.unsqueeze(1),
            done=done.unsqueeze(1),
            target_qvalue=q_t.unsqueeze(1),
            target_action=q_t.argmax(-1).long().unsqueeze(1),
            gamma=self._discount,
            n_step=self._n_step,
            eps=self._rescale_epsilon,
            eta=self._priority_eta,
        )

        return prioritiy.item()

    def _update_target_network(self):
        self._target_network.load_state_dict(self._online_network.state_dict())

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'discount': self._discount,
            'updates': self._update_t,
        }
