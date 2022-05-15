from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from agents.networks.dqn import DqnConvNet
from agents.dqn import agent
from agents.checkpoint import PyTorchCheckpoint
from agents.schedule import LinearSchedule
from agents import main_loop
from agents import gym_env
from agents import greedy_actors
from agents import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('replay_capacity', 50000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 5000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 32, 'Sample batch size when do learning.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.05, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_decay_step', 500000, 'Total steps to decay value of the exploration rate.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('n_step', 3, 'TD n-step bootstrap.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 10, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e6), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(1e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer('learn_frequency', 4, 'The frequency (measured in agent steps) to do learning.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    10000,
    'The frequency (measured in number of online Q network parameter updates) to update target Q networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/dqn_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/dqn', 'Path for checkpoint directory.')


def main(argv):
    """Trains DQN agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder(random_int=0):
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=FLAGS.seed + int(random_int),
            noop_max=30,
            done_on_life_loss=True,
            clip_reward=True,
        )

    env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', env.action_space.n)
    logging.info('Observation spec: %s', env.observation_space.shape)

    input_shape = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
    num_actions = env.action_space.n

    # Test environment and state shape.
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == input_shape

    network = DqnConvNet(input_shape=input_shape, num_actions=num_actions)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    q = network(torch.from_numpy(obs[None, ...])).q_values
    assert q.shape == (1, num_actions)

    # Create e-greedy exploration epsilon schdule
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create transition replay
    replay = replay_lib.UniformReplay(
        capacity=FLAGS.replay_capacity, structure=replay_lib.TransitionStructure, random_state=random_state
    )

    # Create DQN agent instance
    train_agent = agent.Dqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.NStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_frequency=FLAGS.learn_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        n_step=FLAGS.n_step,
        discount=FLAGS.discount,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        num_actions=num_actions,
        random_state=random_state,
        device=runtime_device,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        name='DQN-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(FLAGS.checkpoint_path)
    state = checkpoint.state
    state.environment_name = FLAGS.environment_name
    state.iteration = 0
    state.network = network

    # Run the traning and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        network=network,
        train_agent=train_agent,
        train_env=env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        tensorboard=FLAGS.tensorboard,
        tag=FLAGS.tag,
        max_episode_steps=FLAGS.max_episode_steps,
    )


if __name__ == '__main__':
    app.run(main)
