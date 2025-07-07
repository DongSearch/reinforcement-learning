import copy
from typing import Any, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]


        # TODO: Define your action_space and observation_space here
        # we have four action(U,D,L,R)
        self.action_space = spaces.Discrete(4)
        # it is for the state having inter value in range between 0-4 with 2 dimention
        self.observation_space = spaces.Box(low=0,high=4,shape=(2,),dtype=np.int32)

        self.agent_position = [0, 0]

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # TODO: Write your implementation here
        # reset the position of agent
        self.agent_position=[0,0]

        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = None
        done = None

        # TODO: Write your implementation here
        match action:
            case 0 : # up
                self.agent_position[0] -=1
            case 1 : # right
                self.agent_position[1] +=1
            case 2 : # down
                self.agent_position[0] +=1
            case 3 : # left
                self.agent_position[1] -=1
        # not to exceed the range, we need to clip it
        self.agent_position[0] = clamp(self.agent_position[0],0,3)
        self.agent_position[1] = clamp(self.agent_position[1], 0, 3)
        observation = self._observe()
        reward = 0
        done = False

        if 't' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward = -1
            done = True
        if 'g' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward = +1
            done = True


        return observation, reward, done, False, {}

    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass

    def _observe(self):
        return np.array(self.agent_position)
