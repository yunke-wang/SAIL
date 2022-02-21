from my_utils import *
from args_parser import *
from core.agent import Agent

from core.dqn import *
from core.ac import *
from core.irl import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import math
""" The main entry function for RL """


def main(args):
    if args.il_method is None:
        method_type = "RL"  # means we just do RL with environment's rewards
        info_method = False
        encode_dim = 0
    else:
        method_type = "IL"
        if "info" in args.il_method:
            info_method = True
            encode_dim = args.encode_dim
        else:
            info_method = False
            encode_dim = 0

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = True  # Set to True to avoid moving gym's state to gpu tensor every step during testing.

    env_name = args.env_name
    """ Create environment and get environment's info. """
    env = gym.make(env_name)
    env.seed(args.seed)
    env_test = gym.make(env_name)
    env_test.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    """ always normalize env. """
    if np.asscalar(env.action_space.high[0]) != 1:
        from my_utils.my_gym_utils import NormalizeGymWrapper
        env = NormalizeGymWrapper(env)
        env_test = NormalizeGymWrapper(env_test)
        print("Use state-normalized environments.")
    a_bound = np.asscalar(env.action_space.high[0])
    a_low = np.asscalar(env.action_space.low[0])
    assert a_bound == -a_low
    assert a_bound == 1
    print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """define actor and critic"""
    if args.rl_method == "ac":
        policy_updater = AC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound,
                            encode_dim=encode_dim)
    if args.rl_method == "sac":
        policy_updater = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound,
                             encode_dim=encode_dim)
    if args.rl_method == "td3":
        policy_updater = TD3(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound,
                             encode_dim=encode_dim)
    if args.rl_method == "trpo":
        policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound,
                              encode_dim=encode_dim)
    if args.rl_method == "ppo":
        policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound,
                             encode_dim=encode_dim)

    update_type = policy_updater.update_type  # "on_policy" or "off_policy"
    if args.max_step is None:
        if update_type == "on_policy":
            args.max_step = 5000000
        elif update_type == "off_policy":
            args.max_step = 1000000

    if method_type == "IL":
        if args.il_method == "irl":
            discriminator_updater = IRL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "gail":
            discriminator_updater = GAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "vail":
            discriminator_updater = VAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "airl":
            discriminator_updater = AIRL(state_dim=state_dim, action_dim=action_dim, args=args,
                                         policy_updater=policy_updater)
        elif args.il_method == "infogail":
            discriminator_updater = InfoGAIL(state_dim=state_dim, action_dim=action_dim, args=args,
                                             policy_updater=policy_updater)
        elif args.il_method == "sail":
            discriminator_updater = SAIL(state_dim=state_dim, action_dim=action_dim, args=args)

        elif args.il_method == "iwil":
            discriminator_updater = IWIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "icgail":
            discriminator_updater = ICGAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif 'rex' in args.il_method:
            discriminator_updater = REX(state_dim=state_dim, action_dim=action_dim, args=args)

    """ Set method and hyper parameter in file name"""
    if method_type == "RL":
        method_name = args.rl_method.upper()
        hypers = rl_hypers_parser(args)
    else:
        method_name = args.il_method.upper() + "_" + args.rl_method.upper()
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)

        if args.il_method == "infogail" and args.info_loss_type.lower() != "bce":
            method_name += "_" + args.info_loss_type.upper()

        if args.soft == True:
            method_name = method_name + '_Soft'


    if method_type == "RL":
        exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)
    elif method_type == "IL":
        exp_name = "%s-%s-%s_s%d" % (discriminator_updater.traj_name, method_name, hypers, args.seed)

    """ Set path for result and model files """
    result_path = "./results_%s/%s/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name)
    model_path = "./results_%s/%s_models/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name)
    pathlib.Path("./results_%s/%s/%s" % (method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True)
    pathlib.Path("./results_%s/%s_models/%s" % (method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True)
    print("Running %s" % (colored(method_name, p_color)))
    print("%s result will be saved at %s" % (colored(method_name, p_color), colored(result_path, p_color)))
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(result_path)

    """ Function to update the parameters of value and policy networks"""
    def update_params_g(batch):
        states = torch.FloatTensor(np.stack(batch.state)).to(device)
        next_states = torch.FloatTensor(np.stack(batch.next_state)).to(device)
        masks = torch.FloatTensor(np.stack(batch.mask)).to(device).unsqueeze(-1)

        actions = torch.FloatTensor(np.stack(batch.action)).to(device)

        if method_type == "RL":
            rewards = torch.FloatTensor(np.stack(batch.reward)).to(device).unsqueeze(-1)
            policy_updater.update_policy(states, actions.to(device), next_states, rewards, masks)
        elif method_type == "IL":
            nonlocal d_rewards
            if 'rex' in args.il_method:
                d_rewards = discriminator_updater.compute_reward(states,actions,masks=masks).detach().data
            else:
                d_rewards = discriminator_updater.compute_reward(states, actions).detach().data

            # Append one-hot vector of context to state.
            if info_method:
                latent_codes = torch.LongTensor(np.stack(batch.latent_code)).to(device).view(-1, 1)  # [batch_size, 1]
                d_rewards += discriminator_updater.compute_posterior_reward(states, actions, latent_codes).detach().data

                latent_codes_onehot = torch.FloatTensor(states.size(0), encode_dim).to(device)
                latent_codes_onehot.zero_()
                latent_codes_onehot.scatter_(1, latent_codes, 1)  # should have size [batch_size, num_worker]

                states = torch.cat((states, latent_codes_onehot), 1)
                next_states = torch.cat((next_states, latent_codes_onehot), 1)

            policy_updater.update_policy(states, actions, next_states, d_rewards, masks)

    """ Storage and counters """
    memory = Memory(capacity=1000000)  # Memory buffer with 1 million max size.
    step, i_iter, tt_g, tt_d, perform_test = 0, 0, 0, 0, 0
    d_rewards = torch.FloatTensor(1).fill_(0)  ## placeholder
    log_interval = args.max_step // 1000  # 1000 lines in the text files
    save_model_interval = (log_interval * 10)  # * (platform.system() != "Windows")  # do not save model ?
    print("Max steps: %s, Log interval: %s steps, Model interval: %s steps" % \
          (colored(args.max_step, p_color), colored(log_interval, p_color), colored(save_model_interval, p_color)))

    """ Reset seed again """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Agent for testing in a separated environemnt """
    agent_test = Agent(env_test, render=args.render, t_max=args.t_max, test_cpu=test_cpu)

    latent_code = None  ## only for infogail
    state = env.reset()
    done = 1
    """ The actual learning loop"""
    for total_step in tqdm(range(0, args.max_step + 1), dynamic_ncols=True):

        """ Save the learned policy model """
        if save_model_interval > 0 and total_step % save_model_interval == 0:
            policy_updater.save_model("%s_policy_T%d.pt" % (model_path, total_step))
            if 'rex' not in args.il_method:
                discriminator_updater.save_model("%s_reward_T%d.pt" % (model_path, total_step))

        """ Test the policy before update """
        if total_step % log_interval == 0:
            perform_test = 1

        """ Test learned policy """
        if perform_test:
            if not info_method:
                log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                perform_test = 0
            else:
                log_test = []
                for i_k in range(0, encode_dim):
                    # latent_code_test = discriminator_updater.sample_code().fill_(i_k)   # legacy code that change rng sequences. Use this line to reproduce old results.
                    latent_code_test = torch.LongTensor(size=(1, 1)).fill_(i_k)
                    latent_code_onehot_test = torch.FloatTensor(1, encode_dim)
                    latent_code_onehot_test.zero_()
                    latent_code_onehot_test.scatter_(1, latent_code_test, 1)
                    log_test += [agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10,
                                                                 latent_code_onehot=latent_code_onehot_test.squeeze())]  # use 1 instead of 10 to save time?
                perform_test = 0

        if info_method and latent_code is None:
            latent_code = discriminator_updater.sample_code()  # sample scalar latent code from the prior p(c) which is uniform.
            latent_code_onehot = torch.FloatTensor(1, encode_dim)
            latent_code_onehot.zero_()
            latent_code_onehot.scatter_(1, latent_code, 1)
            latent_code_onehot = latent_code_onehot.squeeze()  # should have size [encode_dim]
            latent_code = latent_code.detach().numpy()

        state_var = torch.FloatTensor(state)
        if latent_code is not None:
            state_var = torch.cat((state_var, latent_code_onehot), 0)

        """ take env step """
        if total_step <= args.random_action and update_type == "off_policy":  # collect random actions first for off policy methods
            action = env.action_space.sample()
        else:
            action = policy_updater.sample_action(state_var.to(device).unsqueeze(0)).to(device_cpu).detach().numpy()

        next_state, reward, done, _ = env.step(action)

        if step + 1 == args.t_max:
            done = 1
        memory.push(state, action, int(not done), next_state, reward, latent_code)
        state = next_state
        step = step + 1

        """ reset env """
        if done:  # reset
            state = env.reset()
            step = 0
            latent_code = None

        """ Update policy """
        if update_type == "on_policy":
            if memory.size() >= args.big_batch_size and done:
                batch = memory.sample()

                if method_type == "IL":
                    if (args.il_method == 'sail' and total_step >= args.initial_step and args.weight == True):
                        for i_d in range(0, args.d_step):
                            if args.soft == True:
                                k, weight, index = discriminator_updater.get_soft_weight(total_step)
                            else:
                                k, weight, index = discriminator_updater.get_weight(total_step)
                            t0_d = time.time()
                            w_dist, total_loss = discriminator_updater.update_discriminator_weight(batch=batch, index=index, weight=weight,total_step=total_step)
                            tt_d += time.time() - t0_d
                            writer.add_scalar('Train/K', k, total_step)
                            writer.add_scalar('Train/Pos_num', weight.sum(), total_step)
                    # for infogail --infogail linear weight #
                    elif (args.il_method == 'infogail' and total_step >= args.initial_step and args.weight == True):
                        for i_d in range(0, args.d_step):
                            if args.soft == True:
                                k, weight, index = discriminator_updater.get_soft_weight(total_step)
                            else:
                                k, weight, index = discriminator_updater.get_weight(total_step)
                            t0_d = time.time()
                            w_dist, total_loss = discriminator_updater.update_discriminator_weight(batch=batch, index=index, weight=weight,total_step=total_step)
                            tt_d += time.time() - t0_d
                            writer.add_scalar('Train/K', k, total_step)
                            writer.add_scalar('Train/Pos_num', weight.sum(), total_step)
                            writer.add_scalar('Loss/W_distance', w_dist, total_step)
                            writer.add_scalar('Loss/Loss', total_loss, total_step)

                    # warmup at first 10% interation
                    elif 'rex' not in args.il_method:
                        for i_d in range(0, args.d_step):
                            index = discriminator_updater.index_sampler()
                            t0_d = time.time()
                            if args.il_method == 'sail':
                                w_dist, total_loss = discriminator_updater.update_discriminator(batch=batch, index=index, total_step=total_step)
                            else:
                                discriminator_updater.update_discriminator(batch=batch, index=index, total_step=total_step)
                            tt_d += time.time() - t0_d

                if args.il_method == 'sail':
                    writer.add_scalar('Loss/W_distance', w_dist, total_step)
                    writer.add_scalar('Loss/Loss', total_loss, total_step)
                t0_g = time.time()
                update_params_g(batch=batch)
                tt_g += time.time() - t0_g
                memory.reset()

        elif update_type == "off_policy":
            if total_step >= args.big_batch_size:

                if method_type == "IL":
                    if (args.il_method == 'sail' and total_step >= args.initial_step and args.weight == True):
                        if args.soft == True:
                            k, weight, index, ratio, ratios = discriminator_updater.get_soft_weight(total_step)
                        else:
                            k, weight, index, ratio, ratios = discriminator_updater.get_weight(total_step)
                        t0_d = time.time()
                        w_dist, total_loss = discriminator_updater.update_discriminator_weight(batch=batch, index=index, weight=weight,total_step=total_step)
                        tt_d += time.time() - t0_d
                        writer.add_scalar('Train/K', k, total_step)
                        writer.add_scalar('Train/Pos_num', weight.sum(), total_step)
                        writer.add_scalar('Train/Ratio', ratio, total_step)
                    else:
                        index = discriminator_updater.index_sampler()
                        batch = memory.sample(args.mini_batch_size)
                        t0_d = time.time()
                        discriminator_updater.update_discriminator(batch=batch, index=index, total_step=total_step)
                        tt_d += time.time() - t0_d
                elif method_type == "RL":
                    batch = memory.sample(args.mini_batch_size)

                t0_g = time.time()
                update_params_g(batch=batch)
                tt_g += time.time() - t0_g

        """ Print out result to stdout and save it to a text file for plotting """
        if total_step % log_interval == 0:

            result_text = t_format("Step %7d " % (total_step), 0)
            if method_type == "RL":
                result_text += t_format("(g%2.2f)s" % (tt_g), 1)
            elif method_type == "IL":
                c_reward_list = d_rewards.to(device_cpu).detach().numpy()
                result_text += t_format("(g%2.1f+d%2.1f)s" % (tt_g, tt_d), 1)
                result_text += " | [D] " + t_format("min: %.2f" % np.amin(c_reward_list), 0.5) + t_format(
                    " max: %.2f" % np.amax(c_reward_list), 0.5)

            result_text += " | [R_te] "
            if not info_method:
                result_text += t_format("min: %.2f" % log_test['min_reward'], 1) + t_format(
                    "max: %.2f" % log_test['max_reward'], 1) \
                               + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
                if args.env_id >=1 and args.env_id <=10:
                    result_text += "| Distance: "
                    result_text += t_format(
                        " %.2f (%.2f)" % (log_test['distance'], log_test['distance_std']), 2)
            else:
                result_text += "Avg "
                for i_k in range(0, encode_dim):
                    result_text += t_format(
                        "%d: %.2f (%.2f)" % (i_k, log_test[i_k]['avg_reward'], log_test[i_k]['std_reward']), 2)
                if args.env_id >= 1 and args.env_id <= 10:
                    result_text += "| Distance: "
                    for i_k in range(0, encode_dim):
                        result_text += t_format("%d: %.2f (%.2f)" % (i_k, log_test[i_k]['distance'], log_test[i_k]['distance_std']), 2)

            if (args.rl_method == "sac"):
                result_text += ("| ent %0.3f" % (policy_updater.entropy_coef))

            tt_g = 0
            tt_d = 0
            tqdm.write(result_text)
            with open(result_path + ".txt", 'a') as f:
                print(result_text, file=f)


if __name__ == "__main__":
    args = args_parser()
    main(args)

