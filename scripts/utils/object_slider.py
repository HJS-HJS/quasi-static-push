import numpy as np
from utils.diagram import Diagram

class ObjectSlider(object):
    '''
    2d slider set
    '''
    def __init__(self):
        self.sliders:list[Diagram] = []

    def append(self, item:Diagram): self.sliders.append(item)

    def __len__(self)->int: return len(self.sliders)
    
    def __getitem__(self, i:int)->Diagram: return self.sliders[i]

    def __iter__(self): 
        for slider in self.sliders:
            yield slider

    def apply_q(self, q:np.array):
        _idx = int(len(q) / len(self.sliders))
        for i, slider in enumerate(self.sliders):
            slider.q = q[i * _idx:(i + 1) * _idx]
        
    def apply_v(self, v:np.array):
        _idx = int(len(v) / len(self.sliders))
        for i, slider in enumerate(self.sliders):
            slider.v = v[i * _idx:(i + 1) * _idx]

    @property
    def q(self)->np.array:
        return np.hstack([slider.q for slider in self.sliders]).reshape(-1)
    
    @property
    def v(self)->np.array:
        return np.hstack([slider.v for slider in self.sliders]).reshape(-1)

    @property
    def r(self)->np.array:
        return np.hstack([slider.r for slider in self.sliders]).reshape(-1)