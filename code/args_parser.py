from my_utils import *

def args_parser():
    parser = argparse.ArgumentParser(description='RL and IL cores.')

    parser.add_argument('--env_id', type=int, default=4, help='Id of environment')
    parser.add_argument('--render', type=int, default=0, help='render the environment during testing')
    parser.add_argument('--t_max', type=int, default=1000, help='maximum time step in one episode of each environment')
    parser.add_argument('--pixel_mode', type=int, default=0, help='Use pixel observation? WIP.')
    parser.add_argument('--debug_mode', type=int, default=0, help='Debug/protoptyping mode')

    ## Seeds    
    parser.add_argument('--seed', type=int, default=1, help='random seed for all (default: 1)')
    parser.add_argument('--max_step', type=int, default=None, help='maximal number of steps. (1 Million default)')
    parser.add_argument('--rl_method', default="trpo", help='method to optimize policy')
    parser.add_argument('--il_method', default='sail', help='method to optimize reward')

    ## Network architecuture
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[None], help='list of hidden layers')
    parser.add_argument('--activation', action="store", default="relu", choices=["relu", "tanh", "sigmoid", "leakyrelu"], help='activation function')
 
    ## Step size, batch size, etc
    parser.add_argument('--learning_rate_pv', type=float, default=None, help='learning rate of policy and value')
    parser.add_argument('--learning_rate_d', type=float, default=None, help='learning rate of discriminator ')
    parser.add_argument('--mini_batch_size', type=int, default=256, help='Mini batch size for all updates (256)')
    parser.add_argument('--big_batch_size', type=int, default=1000, help='Big batch size per on-policy update')    
    parser.add_argument('--random_action', type=int, default=10000, help='Nmber of initial random action for off-policy methods (10000)')
    parser.add_argument('--tau_soft', type=float, default=0.005, help='tau for soft update (0.01 or 0.005)')  
    parser.add_argument('--gamma', type=float, default=None, help='discount factor. default: 0.99')
    parser.add_argument('--entropy_coef', default=None, help='Coefficient of entropy bonus')

    ## SAC options
    parser.add_argument('--log_std', type=float, default=0, help='initial log std (default: 0)')
    parser.add_argument('--symmetric', type=int, default=1, help='Use symmetric sampler (antithetic) or not)')
    parser.add_argument('--target_entropy_scale', type=float, default=1, help='Scale of entropy target')

    ## TRPO/PPO common options
    parser.add_argument('--trpo_max_kl', type=float, default=0.01, help='max KL for TRPO')
    parser.add_argument('--trpo_damping', type=float, default=0.1, help='trpo_damping scale of FIM for TRPO')
    parser.add_argument('--gae_tau', type=float, default=0.97, help='lambda for GAE (default: 0.97)')
    parser.add_argument('--gae_l2_reg', type=float, default=0, help='l2-regularization for GAE (default: no regularization)')
    
    ## PPO options
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='Clip epsilon of PPO')
    parser.add_argument('--ppo_early', type=int, default=1, help='Early stop in PPO?')
    parser.add_argument('--ppo_gradient_clip', type=float, default=0, help='gradient clipping in PPO')
    parser.add_argument('--ppo_separate_net', type=int, default=1, help='Use separate nets for policy and value?')

    ## DDPG/TD3 options 
    parser.add_argument('--explore_std', type=float, default=0.1, help='exploration standard deviation')

    ## IRL options     
    parser.add_argument('--d_step', type=int, default=5, help='Number of discriminator update for on-policy methods')
    parser.add_argument('--clip_discriminator', default=None, help='clip reward in (-x, x) by sigmoid?')
    parser.add_argument('--gp_lambda', default=None, help='gradient penalty regularization')
    parser.add_argument('--gp_center', type=float, default=1, help='center of gradient penalty regularization')
    parser.add_argument('--gp_alpha', action="store", default="mix",choices=["mix", "real", "fake"], help='interpolation type of gradient penalty regularization')
    parser.add_argument('--gp_lp', type=int, default=0, help='Use LP version (with max) of gradient penalty regularization?')
    parser.add_argument('--bce_negative', type=int, default=0, help='Use negative reward for gail')

    ## InfoGAIL    
    parser.add_argument('--encode_dim', type=int, default=None, help='Dimension of z ')
    parser.add_argument('--info_coef', type=float, default=0.1, help='coefficient in infogail ')
    parser.add_argument('--info_loss_type', default="bce", help='loss type ("linear", "bce", "ace")')

    ## Demonstration      
    parser.add_argument('--c_data', type=int, default=12, help='Use a preset of dataset')
    parser.add_argument('--demo_iter', type=int, default=None, help='index to expert data')
    parser.add_argument('--demo_file_size', type=int, default=10000, help='file name with max number of the expert state-action pairs (100000)')
    parser.add_argument('--demo_split_k', type=int, default=0, help='Split demonstrations into trajectories each with unique k or not? (0)')
    parser.add_argument('--traj_deterministic', type=int, default=0, help='Use trajectory with determinisitic policy')
    parser.add_argument('--noise_type', action="store", default="normal", choices=["normal", "SDNTN"], help="Noise type (normal: Gaussian noisy policy, SDNTN: TSD noisy policy)")
    parser.add_argument('--noise_level_list', type=float, default=[0.0], help='Noise level of demo to load')

    ## plotting.
    parser.add_argument('--plot_save', type=int, default=0, help='save plot or not.')
    parser.add_argument('--plot_large', type=int, default=1, help='plot large figures.')
    parser.add_argument('--plot_show', type=int, default=1, help='show plot or not.')

    ## SAIL
    parser.add_argument('--initial_step', type=int, default=500000, help='10% warmup steps')
    parser.add_argument('--exp_n', type=int, default=3, help='exp')
    parser.add_argument('--weight', action='store_true')
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--alpha', type=float, default=math.e)
    parser.add_argument('--traj-size', type=int, default=None)

    ## drex
    parser.add_argument('--num_steps', type=int, default=40, help='number of demonstrations for T-REX training')
    parser.add_argument('--drex_noise', default=np.arange(0., 1.0, 0.05), help='generate trajectories with various noises')
    parser.add_argument('--model_path', type=str, default=None, help='pre-trained reward model, which can be acquired by trex_main.py')
    parser.add_argument('--time-index', action='store_true', default=False, help='whether to use time-index imperfect demonstrations')

    args = parser.parse_args()

    """ Using ID instead of env name is more convenient """
    env_dict = {
                # Mujoco
                1 : "InvertedPendulum",
                2 : "HalfCheetah",
                3 : "Reacher",
                4 : "Swimmer",
                5 : "Ant",
                6 : "Hopper",
                7 : "Walker2d",
                8 : "InvertedDoublePendulum",
                9 : "Humanoid",
                10: "HumanoidStandup",
    }
    args.env_name = env_dict[args.env_id]
    args.env_name += "-v2"

    """ learning methods' defaults """
    rl_default_parser(args) 
    irl_default_parser(args)

    # if args.il_method == 'iwil' or 'icgail':
    #     args = conf_hypers_parser(args)

    return args

def rl_default_parser(args):     
    if args.hidden_size[0] is None:
        if args.rl_method == "sac":
            args.hidden_size = (100, 100)
            args.activation = "relu"
        elif args.rl_method == "td3":
            args.hidden_size = (100, 100)
            args.activation = "relu"
        elif args.rl_method == "trpo":
            args.hidden_size = (100, 100)
            args.activation = "tanh"
        elif args.rl_method == "ppo":
            args.hidden_size = (100, 100)
            args.activation = "relu"
        if args.il_method is not None and "bc" in args.il_method:
            args.hidden_size = (100, 100)
            args.activation = "tanh"
            if args.env_id == 9:
                args.activation = "relu"
        
    if args.learning_rate_pv is None:
        args.learning_rate_pv = 3e-4

    if args.learning_rate_d is None:
        args.learning_rate_d = 3e-4

    if args.gamma is None:
        args.gamma = 0.99  #
        if args.rl_method == "trpo":    # 0.99 is okay. Use 0.995 to reproduce old results. 
            args.gamma = 0.995
            
    if args.entropy_coef == "None": args.entropy_coef = None 
    if args.entropy_coef is None:
        if args.rl_method == "trpo":
            args.entropy_coef = 0.0001
        elif args.rl_method == "ppo":
            args.entropy_coef = 0.0001
        else:
            args.entropy_coef = 0.0001
    args.entropy_coef = float(args.entropy_coef) 
            
    if "Reacher" in args.env_name:  # not included custom reacher task in robosuite. 
        args.t_max = 50 

def irl_default_parser(args):   
    if args.il_method == "None": args.il_method = None 

    if args.demo_iter is None:
        if args.rl_method == "sac" or args.rl_method == "td3":
            args.demo_iter = 1000000    # demonstrations from sac
        else:
            args.demo_iter = 5000000    # demonstrations from trpo 

    """ default datasets """
    """ To reproduce results in the paper, please set c_data with 12 and 16. """
    if args.c_data == 1:
        args.noise_level_list = [args.noise_level_list]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
    elif args.c_data == 12: # stage 2
        args.noise_level_list = [0.01, 0.25, 0.4, 0.6]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
    elif args.c_data == 13:  # Ant
        args.noise_level_list = [0.01, 0.1, 0.25, 0.4]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
    elif args.c_data == 16:  # stage 1
        args.noise_level_list = [0.01, 0.25, 0.80, 1.0]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
    else:
        raise NotImplementedError

    if args.time_index == True:
        args.noise_level_list = [1000000, 3000000, 5000000]
        args.demo_file_size = 1000
        args.worker_num = len(args.noise_level_list)
        if args.env_id == 5:
             args.noise_level_list = [1500000, 3000000, 5000000]

    if args.gp_lambda == "None": args.gp_lambda = None 
    if args.gp_lambda is None:
        args.gp_lambda = 10    
        if args.rl_method == "ppo":    
            args.gp_lambda = 0.0 # otherwise it converge too slow. 
    args.gp_lambda = float(args.gp_lambda)      

    if args.clip_discriminator == "None": args.clip_discriminator = None 
    if args.clip_discriminator is None:
        if args.il_method == "irl" or args.il_method == "airl" or args.il_method == "sail":
            args.clip_discriminator = 5     # bound reward in (0, 5) by using sigmoid * 5
        elif args.il_method == "infogail" and args.info_loss_type == "linear":   # Wasserstein version of infogail 
            args.clip_discriminator = 5    
        else:
            args.clip_discriminator = 0    # no clip.
    args.clip_discriminator = float(args.clip_discriminator) 
    
    """ infogail defaults """ 
    if args.encode_dim == "None": args.encode_dim = None 
    if args.encode_dim is None: args.encode_dim = args.worker_num

def rl_hypers_parser(args):
    hypers = ""

    for i in range(len(args.hidden_size)):
        hypers += "%d-" % (args.hidden_size[i])
    hypers += "%s" % (args.activation)

    if args.rl_method != "sac" and args.rl_method != "td3" and "dqn" not in args.rl_method:
        hypers += "_ec%0.5f" % args.entropy_coef
        
    if args.debug_mode:
        hypers = "X_" + hypers

    return hypers 

def irl_hypers_parser(args):
    hypers = ""
    hypers += "gp%0.3f" % args.gp_lambda
    hypers += "_cr%d" % args.clip_discriminator #legacy name is Clip Reward

    if args.bce_negative:   # use negative version of log-sigmoid rewards. 
        if args.il_method == "gail" \
            or args.il_method == "vail" \
            or (args.il_method == "infogail" and args.info_loss_type.lower() == "bce"):
            hypers += "_neg"

    if args.debug_mode:
        hypers = "X_" + hypers

    return hypers 

def bc_hypers_parser(args):    
    hypers = ""
    for i in range(len(args.hidden_size)):
        hypers += "%d-" % (args.hidden_size[i])
    hypers += "%s" % (args.activation)

    weight_decay = 1e-5 # should be set via args, but fixed at this value for now.
    hypers += "_wdecay%0.5f" % weight_decay      

    if args.debug_mode:
        hypers = "X_" + hypers

    return hypers
