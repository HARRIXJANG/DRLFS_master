from gym.envs.registration import register

register(
    id="Enviroments/WorldofParts",
    entry_point="Enviroments.envs:WorldofPartsEnv",
)

register(
    id="Enviroments/WorldofParts_for_eval",
    entry_point="Enviroments.envs:WorldofPartsEnv_for_eval",
)
