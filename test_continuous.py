from pathlib import Path
from world.environment import Environment
from world.continuous_env import ContinuousEnv

# Passa direttamente il path, non il Grid
env = Environment(Path("grid_configs/small_grid.npy"), sigma=0.0)

# Wrappalo con ContinuousEnv
cont_env = ContinuousEnv(env)

# Reset e stampa lo stato
state = cont_env.reset(agent_start_pos=(1,1))
print("State type:", type(state))
print("State shape:", state.shape)
print("State values:", state)
print()

# Fai qualche step e stampa
for i in range(3):
    action = 0  # Down
    state, reward, done, info = cont_env.step(action)
    print(f"Step {i+1}: state={state}, reward={reward}, done={done}")