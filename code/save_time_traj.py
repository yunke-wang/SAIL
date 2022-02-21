from my_utils import *
from args_parser import *
from core.agent import Agent
from core.dqn import *
from core.ac import *
import h5py
from numpy.linalg import norm
from itertools import count

p_color = "yellow"
print_color = p_color

""" Load policy model file from pt file (pytorch model), replay policy, and save trajectories in hd5 format. """

def save_traj_from_pt(args):
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Create environment and get environment's info. """
    if args.env_id > 10 and args.env_id <= 20:
        import pybullet
        import pybullet_envs
        pybullet.connect(pybullet.DIRECT)
    env = gym.make(args.env_name)
    env.seed(args.seed)

    if args.render:
        env.render(mode="human")

    state_dim = env.observation_space.shape[0]
    is_disc_action = args.env_discrete
    action_dim = (0 if is_disc_action else env.action_space.shape[0])
    if is_disc_action:
        a_bound = 1
        action_num = env.action_space.n
        print("State dim: %d, action num: %d" % (state_dim, action_num))
    else:
        args.a_bound_pre = np.asscalar(env.action_space.high[0])
        """ always normalize env. """
        from my_utils.my_gym_utils import NormalizeGymWrapper
        env = NormalizeGymWrapper(env)
        a_bound = np.asscalar(env.action_space.high[0])
        a_low = np.asscalar(env.action_space.low[0])
        assert a_bound == -a_low
        print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

    """ Set method and hyper parameter in file name"""
    method_name = args.rl_method.upper()
    hypers = rl_hypers_parser(args)
    exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)

    """ Set path for trajectory files """
    traj_path = "../imitation_data/TRAJ_h5_time_GAIL/%s" % (args.env_name)
    pathlib.Path(traj_path).mkdir(parents=True, exist_ok=True)
    print("%s trajectory will be saved at %s" % (colored(method_name, p_color), colored(traj_path, p_color)))

    """define actor and critic"""
    if args.rl_method == "ac":
        policy_updater = AC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    if args.rl_method == "sac":
        policy_updater = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    if args.rl_method == "td3":
        policy_updater = TD3(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    if args.rl_method == "trpo":
        policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)
    if args.rl_method == "ppo":
        policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound)

    """ Load policy """
    time_index_list = np.arange(0, 5000001, 500000)
    for load_step in time_index_list:
        model_path = "./RL_results/TRPO_models/%s/%s-%s" % (args.env_name, args.env_name, exp_name)
        model_filename = model_path + ("_policy_T%d.pt" % (load_step))

        model_path = "./results_RL/TRPO_models/%s/%s" % (args.env_name, args.env_name)
        model_filename = model_path + ("-traj_type0_N10000-TRPO-100-100-tanh_ec0.00010_gp10.000_cr0_s1_policy_T%d.pt" % (load_step))
        policy_updater.load_model(model_filename)
        policy_updater.policy_to_device(device_cpu)
        print("RL model is loaded from %s" % model_filename)

        if use_gpu:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        env.seed(args.seed)

        """ Storages and counters """
        expert_state_list = []
        expert_action_list = []
        expert_mask_list = []
        expert_reward_list = []
        total_step, avg_reward_episode = 0, 0

        for i_episode in count():
            state = env.reset()
            reward_episode = 0

            for t in range(0, args.t_max):

                state_var = torch.FloatTensor(state)

                if args.traj_deterministic:
                    action = policy_updater.greedy_action(torch.FloatTensor(state).unsqueeze(0)).to(
                        device_cpu).detach().numpy()
                else:
                    action = policy_updater.sample_action(torch.FloatTensor(state).unsqueeze(0)).to(
                        device_cpu).detach().numpy()

                # action = np.clip(action, a_min=a_low, a_max=a_bound)

                next_state, reward, done, _ = env.step(np.clip(action, a_min=a_low, a_max=a_bound))

                if args.render:
                    env.render("human")
                    time.sleep(0.001)

                if t + 1 == args.t_max:
                    done = 1

                expert_state_list.append(state)
                expert_action_list.append(action)
                expert_mask_list.append(int(not done))
                expert_reward_list.append(reward)

                state = next_state
                total_step += 1
                reward_episode += reward

                if done:
                    break

            avg_reward_episode += reward_episode

            if i_episode % 10 == 0:
                print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

            if total_step >= args.demo_file_size:
                break

        expert_states = np.array(expert_state_list)
        expert_actions = np.array(expert_action_list)
        expert_masks = np.array(expert_mask_list)
        expert_rewards = np.array(expert_reward_list)

        print("Total steps %d, total episode %d, AVG reward: %f" % (
        total_step, i_episode + 1, avg_reward_episode / (i_episode + 1)))

        print(expert_actions.min())
        print(expert_actions.max())

        """ save data """
        traj_filename = traj_path + (
                    "/%s_TRAJ-N%d_t%d" % (args.env_name, args.demo_file_size, load_step))
        if args.traj_deterministic:
            traj_filename += "_det"

        hf = h5py.File(traj_filename + ".h5", 'w')
        hf.create_dataset('expert_source_path', data=model_filename)  # network model file.
        hf.create_dataset('expert_states', data=expert_states)
        hf.create_dataset('expert_actions', data=expert_actions)
        hf.create_dataset('expert_masks', data=expert_masks)
        hf.create_dataset('expert_rewards', data=expert_rewards)

        print("TRAJ result is saved at %s" % traj_filename)


if __name__ == "__main__":
    args = args_parser()
    save_traj_from_pt(args)
