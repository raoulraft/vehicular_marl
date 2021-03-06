import supersuit
import wandb
import torch as th
from wandb.integration.sb3 import WandbCallback
import os
from stable_baselines3 import PPO, A2C, TD3, SAC, DQN, HerReplayBuffer, HER, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from uac_parallel_env import parallel_env
from result_buffer import ResultBuffer
from supersuit import frame_stack_v1, normalize_obs_v0, vectorize_aec_env_v0
from input_config import InputConfig
from schedule import linear_schedule

os.environ["WANDB_SILENT"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # quick fix for libiomp5md bug, can skip in ubuntu

project_name = "multi-agent-vehicular-parallel"

mu_p = 0.8
mu_o = 4.5
algo = PPO
n_uavs = 8
eval_episodes = 100

alg = "MULTIAGENT"

res_buffer = ResultBuffer(min_n_drone=n_uavs, max_n_drone=n_uavs, min_mu=mu_p, max_mu=mu_p, step_mu=0.5,
                          net_slice=1, change_processing=True, alg=alg)
for consumption_w in [10, 20, 50, 100, 150]:
    config = {"algo": algo if alg == "MULTIAGENT" else alg,
              "n_cpus": 10,
              "uavs": n_uavs,
              "frame_stack": 4,
              "processing_rate": mu_p,
              "offloading_rate": mu_o,
              "transition_probability_low": 1 / 180,
              "transition_probability_high": 1 / 60,
              "shifting_probs": False,  # True gives better learning curve, while False gives nice results quicker
              "lambda_low": 0.5,
              "lambda_high": 2.5,
              "policy_type": "MlpPolicy",
              "total_timesteps": 300000000,  # 1000000
              "max_time": 10000,  # 50000
              "linear_schedule": True,
              }

    input_config = InputConfig(uavs=config["uavs"],
                               frame_stack=config["frame_stack"],
                               processing_rate=config["processing_rate"],
                               offloading_rate=config["offloading_rate"],
                               delay_w=1,
                               consumption_w=consumption_w,
                               lmbda=[config["lambda_low"], config["lambda_high"]],
                               prob_trans=[config["transition_probability_low"],
                                           config["transition_probability_high"]],
                               shifting_probs=config["shifting_probs"],
                               algorithm=alg,
                               max_time=config["max_time"]
                               )
    input_config.print_settings()

    uav_env = parallel_env(input_c=input_config)
    uav_env = supersuit.pettingzoo_env_to_vec_env_v1(uav_env)

    uav_env = supersuit.concat_vec_envs_v1(uav_env, 1, num_cpus=0,
                                           base_class='stable_baselines3')

    eval_env = parallel_env(input_c=input_config, result_buffer=res_buffer)
    eval_env = supersuit.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = supersuit.concat_vec_envs_v1(eval_env, 1, num_cpus=0,
                                            base_class='stable_baselines3')

    # uav_env = supersuit.normalize_obs_v0(uav_env, env_min=0, env_max=1)
    # uav_env = supersuit.frame_stack_v1(uav_env, 3)

    """
    model = PPO('MlpPolicy', uav_env, verbose=0, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
            vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
    """
    # need to initialize it even for static policies to use evaluate_policy from sb3
    model = algo(config["policy_type"],
                 uav_env,
                 verbose=0,
                 gamma=0.99,
                 tensorboard_log=f"runs",
                 learning_rate=linear_schedule(3e-4) if config["linear_schedule"] is True else 3e-4,

                 )

    if alg == "MULTIAGENT":
        print("initializing training")
        # for i in range(3):
        run = wandb.init(
            project=project_name,
            tags=["n {}".format(config["uavs"]),
                  "pr {}".format(config["processing_rate"]),
                  "or {}".format(config["offloading_rate"]),
                  "lmbda_l {:.2f}, lmbda_h {:.2f}".format(config["lambda_low"], config["lambda_high"]),
                  "prob_l {:.2f}, prob_h {:.2f}".format(config["transition_probability_low"],
                                                        config["transition_probability_high"]),
                  "alg {}".format(alg),
                  "weight{}".format(consumption_w),
                  "linear_schedule {}".format(config["linear_schedule"]),
                  ],
            entity="xraulz",
            reinit=True,
            sync_tensorboard=True,
            config=config,
        )
        if os.name == 'nt':
            model.learn(
                total_timesteps=config["total_timesteps"],
            )
        else:
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{project_name}/{run.name}",
                    verbose=2,
                ),
            )
        # model is already saved via model_save_path inside model.learn
        # model.save("policy {}-{}".format(run.name, i))
        run.finish()
    """
    print("initializing evaluation...")
    run = wandb.init(
        project=project_name,
        tags=["n {}".format(config["uavs"]),
              "pr {}".format(config["processing_rate"]),
              "or {}".format(config["offloading_rate"]),
              "lmbda_l {:.2f}, lmbda_h {:.2f}".format(config["lambda_low"], config["lambda_high"]),
              "prob_l {:.2f}, prob_h {:.2f}".format(config["transition_probability_low"],
                                                    config["transition_probability_high"]),
              "evaluation",
              "alg {}".format(alg),
              "weight{}".format(consumption_w),
              ],
        entity="xraulz",
        reinit=True,
        sync_tensorboard=True,
        config=config,
    )
    
    evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
    run.finish()
    """
