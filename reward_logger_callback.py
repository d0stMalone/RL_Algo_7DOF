import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self):
        df = pd.DataFrame({
            "Episode": np.arange(1, len(self.episode_rewards)+1),
            "Reward": self.episode_rewards
        })
        df.to_csv(self.log_path, index=False)
        print(f"📄 Rewards logged to: {self.log_path}")
