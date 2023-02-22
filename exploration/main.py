import gym

from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "LunarLander-v2"
MODEL_ARCHITECTURE = "PPO"


def train():
    training_env = make_vec_env(ENV_ID, n_envs=16)

    # Instantiate the agent
    model = PPO(
        policy="MlpPolicy",
        env=training_env,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.998,
        gae_lambda=0.99,
        ent_coef=0.01,
        verbose=1,
    )

    # Train the agent
    model.learn(total_timesteps=100000)

    return model


def evaluate(model):
    evaluation_env = gym.make(ENV_ID)
    mean_reward, std_reward = evaluate_policy(model, evaluation_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def upload_to_huggingface_hub(model):
    # Define the HuggingFace Hub repo id
    repo_id = f"zap-thamm/{MODEL_ARCHITECTURE}-{ENV_ID}"
    # Define the commit message for the upload of the model
    commit_message = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

    # Create the evaluation env
    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])

    # package_to_hub(
    #     model=model,
    #     model_name=model_name,
    #     model_architecture=MODEL_ARCHITECTURE,
    #     env_id=ENV_ID,
    #     eval_env=eval_env,
    #     repo_id=repo_id,
    #     commit_message=commit_message,
    # )


if __name__ == "__main__":
    # Train the model
    trained_model = train()

    # Optional: save the model
    model_name = "PPO-LunarLander-v2"
    trained_model.save(model_name)

    # Evaluate the model
    evaluate(trained_model)

    # Upload the model
    upload_to_huggingface_hub(trained_model)
