from rl_framework.util import evaluate, upload_to_huggingface_hub
from rl_framework.agent.stable_baselines import StableBaselinesAgent
from rl_framework.environment.gym_environment import GymEnvironmentWrapper

ENV_ID = "LunarLander-v2"
MODEL_ARCHITECTURE = "PPO"
PARALLEL_ENVIRONMENTS = 32

MODEL_NAME = f"{MODEL_ARCHITECTURE}-{ENV_ID}"
REPO_ID = f"zap-thamm/{MODEL_NAME}_development"
COMMIT_MESSAGE = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training
    environments = [GymEnvironmentWrapper(ENV_ID) for _ in range(PARALLEL_ENVIRONMENTS)]

    # Create agent
    agent = StableBaselinesAgent(environments)

    # Train agent
    agent.train()

    # Optional: Save the model
    # agent.save(file_path=f"{MODEL_NAME}")

    # Evaluate the model
    mean_reward, std_reward = evaluate(agent_to_evaluate=agent, evaluation_environment=environments[0])
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Upload the model
    upload_to_huggingface_hub(
        agent_to_upload=agent,
        evaluation_environment=environments[0],
        model_name=MODEL_NAME,
        model_architecture=MODEL_ARCHITECTURE,
        environment_name=ENV_ID,
        repository_id=REPO_ID,
        commit_message=COMMIT_MESSAGE
    )
