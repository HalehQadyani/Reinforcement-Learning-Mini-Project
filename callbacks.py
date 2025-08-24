
import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SuccessLoggerCallback(BaseCallback):
    """
    Logs per-episode success (1/0) to a CSV file. It looks for 'is_success' in the info dict
    returned by env.step(). If not present, you can provide a `success_fn` to compute success
    from obs / info.
    """

    def __init__(self, out_csv: str = "success_log.csv", verbose=0, success_fn=None):
        super().__init__(verbose)
        self.out_csv = out_csv
        self.success_fn = success_fn
       
        if not os.path.exists(self.out_csv):
            with open(self.out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timesteps", "episode", "success"])

        self.episode = 0
        self.episode_timesteps = 0

    def _on_step(self) -> bool:
        
        self.episode_timesteps += 1

      
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        dones = self.locals.get("dones", None)
        if dones is None:
            return True

        if isinstance(dones, (list, tuple, np.ndarray)):
            for i, done in enumerate(dones):
                if done:
                    info = infos[i] if isinstance(infos, (list, tuple)) else infos
                    success = 0
                   
                    if isinstance(info, dict) and info.get("is_success") is not None:
                        success = int(info.get("is_success"))
                    elif self.success_fn is not None:
                        
                        success = int(bool(self.success_fn(info)))
                    
                    with open(self.out_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([self.num_timesteps, self.episode, success])
                    self.episode += 1
        else:
           
            if dones:
                info = infos
                success = 0
                if isinstance(info, dict) and info.get("is_success") is not None:
                    success = int(info.get("is_success"))
                elif self.success_fn is not None:
                    success = int(bool(self.success_fn(info)))
                with open(self.out_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.num_timesteps, self.episode, success])
                self.episode += 1

        return True
