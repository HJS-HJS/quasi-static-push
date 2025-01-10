import numpy as np
from utils.diagram import Diagram

class ObjectObstacle(object):
    def __init__(self):
        self.obstacles:list[Diagram] = []

    def append(self, item:Diagram): self.obstacles.append(item)

    def __len__(self)->int: return len(self.obstacles)
    
    def __getitem__(self, i:int)->Diagram: return self.obstacles[i]

    def __iter__(self): 
        for slider in self.obstacles:
            yield slider

    @property
    def q(self)->np.array:
        return np.hstack([slider.q for slider in self.obstacles]).reshape(-1)
    
    @property
    def v(self)->np.array:
        return np.hstack([slider.v for slider in self.obstacles]).reshape(-1)

    @property
    def r(self)->np.array:
        return np.hstack([slider.r for slider in self.obstacles]).reshape(-1)
    
    def __del__(self):
        for diagram in self:
            del diagram