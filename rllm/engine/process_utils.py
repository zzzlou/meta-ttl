# process_utils.py
import os
from rllm.environments.jericho.openai_helpers import init_global_client

def _worker_reset_wrapper(env, client_config):
    """
    1. Reconnect the OpenAI client, because the global handle is `None` in worker processes.
    2. Execute `env.reset()`.
    """
    pid = os.getpid()
    print(f"🔥 [Worker Process {pid}] Received RESET task for Env {env.idx if hasattr(env, 'idx') else '?'}")
    # `client_config` contains the forwarded `base_url` and `api_key`.
    init_global_client(**client_config)
    
    # Run the normal reset path.
    # At this point `env` is a worker-local copy with its own actor and memory.
    log, info = env.reset()
    
    # Return the updated env together with the reset result.
    return env, log, info

def _worker_step_wrapper(env, action, client_config):
    """
    1. Ensure the OpenAI client is available.
    2. Execute `env.step()`.
    """
    init_global_client(**client_config)
    
    log, reward, done, info = env.step(action)
    
    # Return the updated env so the parent process can observe actor-memory changes.
    return env, log, reward, done, info

def _worker_compute_reward_wrapper(env):
    """Compute the final reward when needed."""
    reward = env.compute_final_reward()
    return env, reward

def _worker_close_wrapper(env):
    """Close the environment."""
    if hasattr(env, 'close'):
        env.close()
    return None
