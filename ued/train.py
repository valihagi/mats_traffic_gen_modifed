import argparse
import os

import numpy as np
import torch
from metadrive import ScenarioEnv
from metadrive.engine import AssetLoader

from cat.saferl_algo import TD3, utils
from cat.saferl_plotter.logger import SafeLogger



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = SafeLogger(
        exp_name=args.mode,
        env_name=args.env,
        seed=args.seed,
        fieldnames=[
            'route_completion_normal',
            'crash_rate_normal',
            'route_completion_adv',
            'crash_rate_adv'
        ]
    )

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    config_train = dict(
        data_directory="datasets/raw_scenes_500",
        start_scenario_index=0,
        num_scenarios=len(os.listdir("datasets/raw_scenes_500")),
        sequential_seed=False,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        no_static_vehicles=True,
        reactive_traffic=False,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    )

    config_test = dict(
        data_directory="datasets/waymo_open_dataset/training",
        crash_vehicle_done=True,
        sequential_seed=True,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        no_static_vehicles=True,
        reactive_traffic=False,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    )

    env = ScenarioEnv(config=config_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.reset(0)
    agent_hparams = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq
    }
    policy = TD3.TD3(**agent_hparams)

    if args.load_model != "":
        policy_file = args.model if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)




    adv_generator.before_episode(env)
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0

    last_eval_step = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        adv_generator.log_AV_history()

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, info = env.step(action)
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        episode_cost += info['cost']

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            adv_generator.after_episode(update_AV_traj=args.mode == 'cat')
            adv_generator.compute_regret()

            print('#' * 20)
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            print(
                f"arrive destination: {info['arrive_dest']} , route_completion: {info['route_completion']}, out of road:{info['out_of_road']}  ")

            # Evaluate episode
            if t - last_eval_step > args.eval_freq:
                last_eval_step = t
                env.close()
                eval_env = WaymoEnv(config=config_test)
                evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv = eval_policy(policy,
                                                                                         eval_env,
                                                                                         adv_generator)
                eval_env.close()
                logger.update([evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv],
                              total_steps=t + 1)

                env = WaymoEnv(config=config_train)

                if args.save_model: policy.save(f"./models/{file_name}")

            # Reset environment
            state, done = env.reset(), False
            # try:
            # 	state, done = env.reset(), False
            # except:
            # 	state, done = env.reset(force_seed=0), False
            # 	print('!!!!!!!!!!!!!Reset Bug!!!!!!!!!!!!!!')
            adv_generator.before_episode(env)

            if args.mode == 'cat' and np.random.random() > max(
                    1 - (2 * t / args.max_timesteps) * (1 - args.min_prob), args.min_prob):
                print('ADVGEN')
                adv_generator.generate()
            else:
                print('NORMAL')

            env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_generator.adv_traj)
            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MDWaymo")
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=10000,
                        type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25000,
                        type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1,
                        type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256,
                        type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise",
                        default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model",
                        default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument('--OV_traj_num', type=int,
                        default=32)  # number of opponent vehicle candidates
    parser.add_argument('--AV_traj_num', type=int,
                        default=5)  # lens of ego traj deque (AV=Autonomous Vehicle is the same as EV(Ego vehcile) in the paper)
    parser.add_argument('--min_prob', type=float,
                        default=0.1)  # The min probability of using raw data in ADV mode
    parser.add_argument('--mode', choices=['replay', 'cat'], \
                        help='Choose a mode (replay, cat)', default='cat')

    args = parser.parse_args()
    main(args)
