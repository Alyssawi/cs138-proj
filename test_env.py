from gymnasium.wrappers import AtariPreprocessing

from curriculum_learning_env import CurriculumLearningEnv

env = CurriculumLearningEnv("ALE/Galaxian-v5", frameskip=1, render_mode="human")

env = AtariPreprocessing(
    env,
    frame_skip=2,
    terminal_on_life_loss=True,
)

obs, info = env.reset()

while True:
    # NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    try:
        action = int(input("Enter Action: "))
        if action > 5:
            continue
    except ValueError:
        continue
    for _ in range(20):
        obs, reward, done, _, _ = env.step(0)

    obs, reward, done, _, _ = env.step(action)

    if done:
        obs, info = env.reset()
