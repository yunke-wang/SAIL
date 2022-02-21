from my_utils import *
from args_parser import *
from core.agent import Agent

from core.ac import *
from core.irl import *

import torch.nn as nn
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, args, state_dim, action_dim, hidden_size=256):
        super(Net, self).__init__()
        self.cuda = args.use_gpu
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim+self.action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)


    def cum_return(self, traj):
        sum_rewards = 0
        sum_abs_reward = 0
        x = F.relu(self.fc1(traj))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = torch.sigmoid(x) * 5
        sum_rewards += torch.sum(x)
        return sum_rewards

    def single_reward(self, traj):
        x = F.relu(self.fc1(traj))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = torch.sigmoid(x)*5
        return x

    def forward(self, traj_i, traj_j):
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0),cum_r_j.unsqueeze(0)), 0)


def load_demo_list(args, verbose=True):
    index_worker = []
    index_worker_idx = []
    m_return_list = []
    index_start = 0
    expert_state_list, expert_action_list, expert_reward_list, expert_mask_list, worker_id_list = [], [], [], [], []

    k_spec_list = args.noise_level_list
    traj_path = "../imitation_data/TRAJ_h5/%s" % (args.env_name)

    """ trajectory description in result files """
    traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
    if args.demo_split_k:
        traj_name = "traj_type%dK_N%d" % (args.c_data, args.demo_file_size)

    if args.noise_type != "normal":
        traj_name += ("_%s" % args.noise_type)
    if args.c_data == 1:
        traj_name += ("_%0.2f" % k_spec_list[0])


    worker_id = 0
    for k in range(0, len(k_spec_list)):
        k_spec = k_spec_list[k]

        traj_filename = traj_path + (
                    "/%s_TRAJ-N%d_%s%0.2f" % (args.env_name, args.demo_file_size, args.noise_type, k_spec))

        if args.traj_deterministic:
            traj_filename += "_det"
        hf = h5py.File(traj_filename + ".h5", 'r')

        expert_mask = hf.get('expert_masks')[:]
        expert_mask_list += [expert_mask][:]
        expert_state_list += [hf.get('expert_states')[:]]
        expert_action_list += [hf.get('expert_actions')[:]]
        expert_reward_list += [hf.get('expert_rewards')[:]]
        reward_array = hf.get('expert_rewards')[:]
        step_num = expert_mask.shape[0]

        ## Set k=n and K=N. Work and pendulum and lunarlander. The results are not included in the paper.
        if not args.demo_split_k:
            worker_id = k
            worker_id_list += [np.ones(expert_mask.shape) * worker_id]
            index_worker_idx += [index_start + np.arange(0, step_num)]
            index_start += step_num

        else:
            ## need to loop through demo until mask = 0, then increase the k counter.
            ## find index in expert_mask where value is 0
            zero_mask_idx = np.where(expert_mask == 0)[0]
            prev_idx = -1
            for i in range(0, len(zero_mask_idx)):
                worker_id_list += [np.ones(zero_mask_idx[i] - prev_idx) * worker_id]
                index_worker_idx += [index_start + np.arange(0, zero_mask_idx[i] - prev_idx)]
                index_start += zero_mask_idx[i] - prev_idx

                worker_id = worker_id + 1
                prev_idx = zero_mask_idx[i]

        traj_num = step_num - np.sum(expert_mask)
        m_return = np.sum(reward_array) / traj_num

        m_return_list += [m_return]

        if verbose:
            print("TRAJ is loaded from %s with traj_num %s, data_size %s steps, and average return %s" % \
                  (colored(traj_filename, p_color), colored(traj_num, p_color), colored(expert_mask.shape[0], p_color), \
                   colored("%.2f" % (m_return), p_color)))

    return expert_state_list, expert_action_list, expert_reward_list, expert_mask_list


def get_demonstrations(args, states, actions, rewards, masks):
    " prepare training data "
    expert_demon = []
    for i in range(1000):
        idx = np.random.choice(len(states), 2, replace=False)
        idx_high = np.argmax(idx)
        idx_min = np.argmin(idx)

        states_high = states[idx_high]
        states_low = states[idx_min]
        actions_high = actions[idx_high]
        actions_low = actions[idx_min]

        start_high = np.random.randint(0, states_high.shape[0] - args.num_steps)
        start_low = np.random.randint(0, states_low.shape[0] - args.num_steps)

        states_0 = states_high[start_high:start_high + args.num_steps, :]
        states_1 = states_low[start_low:start_low + args.num_steps, :]
        actions_0 = actions_high[start_high:start_high + args.num_steps, :]
        actions_1 = actions_low[start_low:start_low + args.num_steps, :]

        demon_0 = np.concatenate((states_0, actions_0), 1)
        demon_1 = np.concatenate((states_1, actions_1), 1)
        expert_demon.append((demon_0, demon_1))

    return expert_demon


class REX():
    def __init__(self, args, device, state_dim, action_dim, env):
        self.env_name = args.env_name
        self.noise_level_list = args.noise_level_list
        self.c_data = args.c_data
        self.demo_file_size = args.demo_file_size
        self.demo_split_k = args.demo_split_k
        self.env = env
        self.noise_type = args.noise_type

        self.expert_state_list = []
        self.expert_action_list = []
        self.expert_reward_list = []
        self.expert_mask_list = []
        self.worker_id_list = []

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.D = []
        self.reward = Net(args, self.state_dim, self.action_dim).to(self.device)

    def load_data(self):
        index_worker = []
        index_worker_idx = []
        m_return_list = []
        index_start = 0


        k_spec_list = self.noise_level_list
        traj_path = "../imitation_data/TRAJ_h5/%s" % (self.env_name)

        """ trajectory description in result files """
        traj_name = "traj_type%d_N%d" % (self.c_data, self.demo_file_size)
        if self.demo_split_k:
            traj_name = "traj_type%dK_N%d" % (self.c_data, self.demo_file_size)

        if self.noise_type != "normal":
            traj_name += ("_%s" % args.noise_type)
        if self.c_data == 1:
            traj_name += ("_%0.2f" % k_spec_list[0])

        worker_id = 0
        for k in range(0, len(k_spec_list)):
            k_spec = k_spec_list[k]

            traj_filename = traj_path + (
                    "/%s_TRAJ-N%d_%s%0.2f" % (self.env_name, self.demo_file_size, self.noise_type, k_spec))

            hf = h5py.File(traj_filename + ".h5", 'r')

            expert_mask = hf.get('expert_masks')[:]
            self.expert_mask_list += [expert_mask][:]
            self.expert_state_list += [hf.get('expert_states')[:]]
            self.expert_action_list += [hf.get('expert_actions')[:]]
            self.expert_reward_list += [hf.get('expert_rewards')[:]]
            reward_array = hf.get('expert_rewards')[:]
            step_num = expert_mask.shape[0]

            ## Set k=n and K=N. Work and pendulum and lunarlander. The results are not included in the paper.
            if not self.demo_split_k:
                worker_id = k
                self.worker_id_list += [np.ones(expert_mask.shape) * worker_id]
                index_worker_idx += [index_start + np.arange(0, step_num)]
                index_start += step_num
            else:
                ## need to loop through demo until mask = 0, then increase the k counter.
                ## find index in expert_mask where value is 0
                zero_mask_idx = np.where(expert_mask == 0)[0]
                prev_idx = -1
                for i in range(0, len(zero_mask_idx)):
                    self.worker_id_list += [np.ones(zero_mask_idx[i] - prev_idx) * worker_id]
                    index_worker_idx += [index_start + np.arange(0, zero_mask_idx[i] - prev_idx)]
                    index_start += zero_mask_idx[i] - prev_idx

                    worker_id = worker_id + 1
                    prev_idx = zero_mask_idx[i]

            traj_num = step_num - np.sum(expert_mask)
            m_return = np.sum(reward_array) / traj_num

            m_return_list += [m_return]

            print("TRAJ is loaded from %s with traj_num %s, data_size %s steps, and average return %s" % \
                      (colored(traj_filename, p_color), colored(traj_num, p_color),
                       colored(expert_mask.shape[0], p_color), \
                       colored("%.2f" % (m_return), p_color)))

    def get_traj(self, steps=40):
        # collect 10000 * D=(a1,a2) a1 better than a2
        num_expert = len(self.expert_state_list)
        GT_preference = []

        for j in range(5000):
            idx1, idx2 = np.random.choice(num_expert, 2, replace=False)
            D_state_1 = self.expert_state_list[idx1]
            D_state_2 = self.expert_state_list[idx2]
            D_action_1 = self.expert_action_list[idx1]
            D_action_2 = self.expert_action_list[idx2]

            r1 = self.expert_reward_list[idx1]
            r2 = self.expert_reward_list[idx2]

            p_idx1 = np.random.randint(D_state_1.shape[0]-steps)
            p_idx2 = np.random.randint(D_state_2.shape[0]-steps)

            r1_part = r1[p_idx1:p_idx1+steps]
            r2_part = r2[p_idx2:p_idx2+steps]

            self.D.append((np.concatenate((D_state_1[p_idx1:p_idx1+steps],D_action_1[p_idx1:p_idx1+steps]),axis=1),
                           np.concatenate((D_state_2[p_idx2:p_idx2+steps],D_action_2[p_idx2:p_idx2+steps]),axis=1)
                           , 0 if idx1 < idx2 else 1))  # D a,b,sigma   sigma=0 indicates a is better

            GT_preference.append(0 if np.sum(r1_part) > np.sum(r2_part) else 1)

        _, _, preference = zip(*self.D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)

        print('Prepare training data......Precise: {:.2f}'.format(np.count_nonzero(preference==GT_preference)/len(GT_preference)))

    def rex_train(self, batch_size=64, l2_reg=0.01, debug=True):
        D_train = self.D[:int(0.8*len(self.D))]
        D_test = self.D[int(0.8*len(self.D)):]

        optimizer = torch.optim.Adam(self.reward.parameters(), 1e-4)
        reward_loss = torch.nn.CrossEntropyLoss()

        for j in tqdm(range(3000), dynamic_ncols=True):
            batch = []
            idxes = np.random.choice(len(D_train), batch_size, replace=False)
            weight_norm = 0

            for i in idxes:
                batch.append(D_train[i])

            b_x, b_y, b_l = zip(*batch)
            x_split = [len(x) for x in b_x]
            y_split = [len(y) for y in b_y]
            b_x, b_y, b_l = np.concatenate(b_x, axis=0), np.concatenate(b_y, axis=0), np.array(b_l)

            b_x = torch.FloatTensor(b_x).to(self.device)
            b_y = torch.FloatTensor(b_y).to(self.device)
            b_label = torch.from_numpy(b_l).to(self.device)

            reward_x = self.reward.single_reward(b_x)
            reward_y = self.reward.single_reward(b_y)

            cum_reward_x = torch.stack([torch.sum(rs_x) for rs_x in torch.split(reward_x, x_split)])
            cum_reward_y = torch.stack([torch.sum(rs_y) for rs_y in torch.split(reward_y, y_split)])

            cum_reward = torch.stack((cum_reward_x, cum_reward_y), 1)

            optimizer.zero_grad()
            loss = reward_loss(cum_reward, b_label)

            # L2 norm
            params = self.reward.parameters()
            for param in params:
                weight_norm += torch.norm(param, 2)
            l2_loss = l2_reg * weight_norm

            loss += l2_loss
            loss.backward()
            optimizer.step()

            if debug and j % 100 == 0:
                pred = torch.where(cum_reward_x > cum_reward_y, torch.full_like(cum_reward_x, 0),
                                   torch.full_like(cum_reward_x, 1))
                train_acc = torch.where(pred == b_label.float(), torch.full_like(pred, 1),
                                        torch.full_like(pred, 0))
                train_acc = train_acc.mean()

                batch_test = []
                idxes = np.random.permutation(len(D_test))
                for i in idxes:
                    batch_test.append(D_test[i])

                b_x, b_y, b_l = zip(*batch_test)
                x_split = [len(x) for x in b_x]
                y_split = [len(y) for y in b_y]
                b_x, b_y, b_l = np.concatenate(b_x, axis=0), np.concatenate(b_y, axis=0), np.array(b_l)
                with torch.no_grad():
                    b_x = torch.FloatTensor(b_x).to(self.device)
                    b_y = torch.FloatTensor(b_y).to(self.device)
                    b_label = torch.FloatTensor(b_l).to(self.device)

                    reward_x = self.reward.single_reward(b_x)
                    reward_y = self.reward.single_reward(b_y)

                    cum_reward_x = torch.stack([torch.sum(rs_x) for rs_x in torch.split(reward_x, x_split)])
                    cum_reward_y = torch.stack([torch.sum(rs_y) for rs_y in torch.split(reward_y, y_split)])

                    pred = torch.where(cum_reward_x > cum_reward_y, torch.full_like(cum_reward_x, 0), torch.full_like(cum_reward_x, 1))
                    acc = torch.where(pred == b_label.float(), torch.full_like(pred,1), torch.full_like(pred,0))
                    acc = acc.mean()

                tqdm.write('Iter: {}, Loss: {:.2f}, Train_acc: {:.2f}, Valid_acc: {:.2f}'.format(j, loss.cpu().detach().numpy(),
                                                        train_acc.cpu().detach().numpy(), acc.cpu().detach().numpy()))

            # if train_acc >= 0.97 and acc >= 0.97:
            #     break

class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Noisepolicy():
    def __init__(self, env, policy, noise_level, noise_type='epsilon'):
        self.policy = policy
        self.env = env
        self.action_space = env.action_space
        self.noise = noise_type

        if self.noise == 'normal':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = NormalActionNoise(mu, std)
        elif self.noise == 'epsilon':
            self.eplison = noise_level
            self.scale = self.action_space.high[0]
        else:
            assert False

    def act(self, obs):
        if self.noise == 'epsilon':
            if np.random.random() < self.eplison:
                return np.random.uniform(-self.scale, self.scale, self.action_space.shape)
            else:
                # act = self.policy.greedy_action(obs).numpy()
                act,_, _ = self.policy(obs)
                act = act.detach().squeeze(0).numpy()
        else:
            act = self.policy.sample_action(obs)
            act += self.action_noise()

        return np.clip(act, self.env.action_space.low, self.env.action_space.high)

    def reset(self):
        self.action_noise.reset()


class Drex(REX):
    def __init__(self, args, device, state_dim, action_dim, env, a_bound):
        super().__init__(args, device, state_dim, action_dim, env)
        self.trajs = []
        self.a_bound = a_bound
        self.weight_decay = 1e-5

    def bc_train(self, args, expert_traj, wgail=1):
        if wgail == 1:
            expert_traj = torch.FloatTensor(expert_traj).to(self.device)
        else:
            expert_state = np.concatenate(self.expert_state_list, axis=0)
            expert_action = np.concatenate(self.expert_action_list, axis=0)
            expert_traj = np.concatenate((expert_state, expert_action), axis=1)
            expert_traj = torch.FloatTensor(expert_traj).to(self.device)

        policy_net = Policy_Gaussian(self.state_dim, self.action_dim,
                                     hidden_size=args.hidden_size, activation=args.activation, param_std=0,
                                     log_std=args.log_std, a_bound=self.a_bound, squash_action=0).to(self.device)

        # policy_net = Policy(self.state_dim, self.action_dim, 100).to(self.device)
        optim_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate_pv,
                                        weight_decay=self.weight_decay)

        iter_num = 5000

        for i in range(iter_num):
            index = torch.randperm(expert_traj.shape[0])[:256]
            expert_state_batch = expert_traj[index, :self.state_dim].to(self.device)
            expert_action_batch = expert_traj[index, self.state_dim:].to(self.device)

            optim_policy.zero_grad()
            action_mean, _, _ = policy_net(expert_state_batch)
            loss = 0.5 * ((expert_action_batch - action_mean) ** 2).mean()
            loss.backward()
            optim_policy.step()
            if i % 500 == 0:
                print('BC step: {}\t Loss: {:.2f}'.format(i, loss.cpu().detach().numpy()))
        print('BC finished.....')

        policy_net = policy_net.cpu()

        r = 0
        for _ in range(10):
            obs = env_test.reset()
            for _ in range(10000):
                action, _, _ = policy_net(torch.FloatTensor(obs).unsqueeze(0))
                obs, reward, done, _ = env_test.step(action.detach().numpy())
                r += reward
                if done:
                    break
        r = r / 10
        print(r)
        return policy_net

    def collect_traj(self, policy, traj_size=1000):
        obs, act, rewards = [self.env.reset()], [], []
        for _ in range(traj_size):
            action = policy.act(torch.FloatTensor(obs[-1]).unsqueeze(0))
            state, reward, done, _ = env.step(action)
            act.append(action)
            rewards.append(reward)
            obs.append(state)

            if done and len(act) < traj_size:
                obs.pop()
                obs.append(self.env.reset())

            if len(obs) == traj_size:
                obs.pop()
                break
        return obs, act, rewards

    def pre_build(self, noise_level, policy):
        trajs = []
        for i in range(noise_level.shape[0]):
            noise_policy = Noisepolicy(self.env, policy, noise_level[i])
            agent_traj = []

            obs, act, rewards = self.collect_traj(noise_policy)
            agent_traj.append((obs, act, rewards))

            trajs.append((noise_level[i], agent_traj))
            print('NoisePolicy:{} finished built'.format(i))
        self.trajs = trajs

    def get_traj(self, steps=40):
        self.D = []
        GT_preference = []
        for i in range(2000):
            idx1, idx2 = np.random.choice(len(self.trajs), 2, replace=False)
            traj1 = self.trajs[idx1][1][0]
            traj2 = self.trajs[idx2][1][0]

            x_ptr = np.random.randint(0, len(traj1[0])-steps)
            y_ptr = np.random.randint(0, len(traj2[0])-steps)

            r1_part = traj1[2][x_ptr:x_ptr+steps]
            r2_part = traj2[2][y_ptr:y_ptr+steps]

            self.D.append((np.concatenate((traj1[0][x_ptr:x_ptr+steps], traj1[1][x_ptr:x_ptr+steps]), axis=1),
                           np.concatenate((traj2[0][y_ptr:y_ptr+steps], traj2[1][y_ptr:y_ptr+steps]), axis=1),
                           0 if self.trajs[idx1][0] < self.trajs[idx2][0] else 1))

            GT_preference.append(0 if np.sum(r1_part) > np.sum(r2_part) else 1)


        _, _, preference = zip(*self.D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)

        print('Prepare training data......Precise: {:.2f}'.format(np.count_nonzero(preference == GT_preference)/len(GT_preference)))


if __name__ == "__main__":

    args = args_parser()
    assert args.il_method == 'trex' or 'drex'
    method_type = "IL"
    torch.manual_seed(args.seed)
    args.use_gpu = True
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = True      # Set to True to avoid moving gym's state to gpu tensor every step during testing.
    env_name = args.env_name
    """ Create environment and get environment's info. """
    env = gym.make(env_name)
    env.seed(args.seed)
    env_test = gym.make(env_name)
    env_test.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    is_disc_action = args.env_discrete
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
    if args.il_method == 'trex':
        trex = REX(args, device, state_dim, action_dim, env)
        trex.load_data()
        trex.get_traj()
        trex.rex_train()
        torch.save(trex.reward.cpu().state_dict(), './reward_model/trex/{}_type{}_reward.pth'.format(env_name, args.c_data))

    elif args.il_method == 'drex':
        drex = Drex(args, device, state_dim, action_dim, env, a_bound)
        traj = drex.load_data(args.stage, args.size)
        traj = torch.FloatTensor(traj).to(device)
        policy_net = drex.bc_train(args, traj, wgail=1)

        drex.pre_build(args.drex_noise, policy_net)
        drex.get_traj()
        drex.rex_train()
        torch.save(drex.reward.cpu().state_dict(), './reward_model/drex/{}_stage{}_reward.pth'.format(env_name, args.stage))

