import tensorflow as tf 
import numpy as np 
from game_state import GameState

class WorkerThread(object):
    def __init__(self, 
                thread_index,
                device,
                random_seed,
                game,
                frame_skip,
                display,
                no_op_max,
                learning_rate,
                final_epsilon,
                gamma,
                optimizer,
                global_network,
                global_max_steps,
                local_max_steps):

        np.random.seed(random_seed)

        self.thread_index = thread_index
        self.local_step = 0
        self.gamma = gamma

        self.global_max_steps = global_max_steps

        self.local_network = ConvolutionalNeuralNetwork(0, device, random_seed + thread_index, learning_rate, optimizer)

        self.game_state = GameState(random_seed, game, frame_skip, display, no_op_max)

        # KANSKE
        #self.sync = self.local_network.sync_from(global_network)

        self.episode_reward = 0

    def train(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []

        # KANSKE
        #sess.run(self.sync)
        
        # Batch update loop:
        local_steps = 0
        while not terminal and local_step < local_max_steps:
            q_values = self.local_network.predict(sess, self.game_state.s_t)
            action = np.argmax(q_values)

            # Fill batch
            states.append(self.game_state.s_t)
            actions.append(action)

            s_t1, reward, terminal = self.game_state.step(action)

            # one-step Q-Learning: add discounted expected future reward
            if not terminal:
                reward = self.gamma * np.max(q_values)
            rewards.append(reward)
            self.episode_reward += reward

            self.local_step += 1

            # Update game state
            self.game_state.update()

            if not terminal






            # Increment shared frame counter
            agent.frame_increment()
            batch_states.append(screen)
            # Exploration vs Exploitation, E-greedy action choose
            if random.random() < epsilon:
                action_index = random.randrange(agent.action_size)
            else:
                reward_per_action = agent.predict_rewards(screen)
                # Choose an action index with maximum expected reward
                action_index = np.argmax(reward_per_action)
            # Execute an action and receive new state, reward for action
            screen, reward, terminal, _ = env.step(action_index)
            reward = np.clip(reward, -1, 1)
            # one-step Q-Learning: add discounted expected future reward
            if not terminal:
                reward += FLAGS.gamma * agent.predict_target(screen)
            batch_rewards.append(reward)
            batch_actions.append(action_index)

            # Increase local steps counter.
            local_steps += 1


        # Apply asynchronous gradient update to shared agent
        agent.train(np.vstack(batch_states), batch_actions, batch_rewards)
        # Anneal epsilon
        epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
        global_epsilons[thread_idx] = epsilon  # Logging


    global global_epsilons
    eps_min = random.choice(EPS_MIN_SAMPLES)
    epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
    print('Thread: %d. Sampled min epsilon: %f' % (thread_idx, eps_min))
    last_logging = agent.frame
    last_target_update = agent.frame
    terminal = True
    # Training loop:
    while agent.frame < FLAGS.total_frames:
        batch_states, batch_rewards, batch_actions = [], [], []
        if terminal:
            terminal = False
            screen = env.reset_random()
        # Batch update loop:
        while not terminal and len(batch_states) < FLAGS.tmax:
            # Increment shared frame counter
            agent.frame_increment()
            batch_states.append(screen)
            # Exploration vs Exploitation, E-greedy action choose
            if random.random() < epsilon:
                action_index = random.randrange(agent.action_size)
            else:
                reward_per_action = agent.predict_rewards(screen)
                # Choose an action index with maximum expected reward
                action_index = np.argmax(reward_per_action)
            # Execute an action and receive new state, reward for action
            screen, reward, terminal, _ = env.step(action_index)
            reward = np.clip(reward, -1, 1)
            # one-step Q-Learning: add discounted expected future reward
            if not terminal:
                reward += FLAGS.gamma * agent.predict_target(screen)
            batch_rewards.append(reward)
            batch_actions.append(action_index)
        # Apply asynchronous gradient update to shared agent
        agent.train(np.vstack(batch_states), batch_actions, batch_rewards)
        # Anneal epsilon
        epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
        global_epsilons[thread_idx] = epsilon  # Logging
        # Logging and target network update
        if thread_idx == 0:
            if agent.frame - last_target_update >= FLAGS.update_interval:
                last_target_update = agent.frame
                agent.update_target()
            if agent.frame - last_logging >= FLAGS.log_interval and terminal:
                last_logging = agent.frame
                saver.save(sess, os.path.join(FLAGS.logdir, "sess.ckpt"), global_step=agent.frame)
                print('Session saved to %s' % FLAGS.logdir)
                episode_rewards, episode_q = test(agent, env, episodes=FLAGS.test_iter)
                avg_r = np.mean(episode_rewards)
                avg_q = np.mean(episode_q)
                avg_eps = np.mean(global_epsilons)
                print("%s. Avg.Ep.R: %.4f. Avg.Ep.Q: %.2f. Avg.Eps: %.2f. T: %d" %
                      (str(datetime.now())[11:19], avg_r, avg_q, avg_eps, agent.frame))
                agent_summary.write_summary({
                    'total_frame_step': agent.frame,
                    'episode_avg_reward': avg_r,
                    'avg_q_value': avg_q,
                    'epsilon': avg_eps
                })
    global training_finished
    training_finished = True
    print('Thread %d. Training finished. Total frames: %s' % (thread_idx, agent.frame))

