import rbc_gym  # noqa: F401
import gymnasium as gym
from timeit import timeit
import time

# initialize julia module
t1 = time.perf_counter(), time.process_time()
env = gym.make(
    "rbc_gym/RayleighBenardConvection2D-v0",
    heater_duration=1,
    render_mode=None,
    use_gpu=True,
)
t2 = time.perf_counter(), time.process_time()
print(f"Julia init time: {t2[0] - t1[0]:.2f} seconds")

# reset env
iterations = 1000
reset_time = timeit("env.reset()", number=iterations, globals=globals())
print(f"Average time to reset env: {reset_time / iterations:.2f} seconds")


# environment step
def step():
    action = env.action_space.sample()
    env.step(action)


step_time = timeit("step()", number=iterations, globals=globals())
print(f"Average time to step one timestep (dt=1): {step_time / iterations:.2f} seconds")
