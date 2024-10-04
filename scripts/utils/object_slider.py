import numpy as np

class ObjectSlider(object):
    '''
    2d slider set
    '''
    def __init__(self):
        self.sliders = []

    def append(self, item): self.sliders.append(item)

    def __len__(self): return len(self.sliders)
    
    def __getitem__(self, i): return self.sliders[i]

    def __iter__(self): 
        for slider in self.sliders:
            yield slider

    def apply_q(self, q):
        _idx = int(len(q) / len(self.sliders))
        for i, slider in enumerate(self.sliders):
            slider.q = q[i * _idx:(i + 1) * _idx]
        
    def apply_v(self, v):
        _idx = int(len(v) / len(self.sliders))
        for i, slider in enumerate(self.sliders):
            slider.v = v[i * _idx:(i + 1) * _idx]

    @property
    def q(self):
        return np.hstack([slider.q for slider in self.sliders]).reshape(-1)
    
    @property
    def v(self):
        return np.hstack([slider.v for slider in self.sliders]).reshape(-1)

    @property
    def r(self):
        return np.hstack([slider.r for slider in self.sliders]).reshape(-1)