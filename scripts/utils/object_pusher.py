import numpy as np
from utils.object_circle import ObjectCircle


class ObjectPusher(object):
    '''
    2d pusher
    '''
    def __init__(self, n_finger, radius, distance, heading, center_x, center_y, rotation):
        self.pushers = []
        self.rel_pose = []
        for i in range(n_finger):
            self.pushers.append(ObjectCircle(radius,
                                             0,
                                             0,
                                             0
                                             ))
            self.rel_pose.append([distance * np.cos(2 * np.pi / n_finger * i + heading),
                                  distance * np.sin(2 * np.pi / n_finger * i + heading),
                                  2 * np.pi / n_finger * i,
                                  ])
        self.radius = radius
        self.q = np.array([center_x, center_y, rotation])
        self.v = np.array([0, 0, 0])
        self.update_pusher()

    def __len__(self): return len(self.pushers)
    
    def __getitem__(self, i): return self.pushers[i]

    def __iter__(self):
        for pusher in self.pushers:
            yield pusher

    def apply_c(self, center):
        self.q[:2] = center
        # self.update_pusher()

    def apply_rot(self, rotation):
        self.q[2] = rotation
        # self.update_pusher()

    def apply_v(self, velocity):
        self.v = velocity
        # self.update_pusher()

    def move_q(self, dq):
        self.q += dq
        self.update_pusher()
    
    def update_pusher(self):
        _rot = np.eye(3)
        _w = np.array([0, 0, self.v[2]])
        _rot[:2,:2] = np.array([
            [np.cos(self.q[2]), np.sin(self.q[2])],
            [np.sin(self.q[2]), -np.cos(self.q[2])],
            ])
        for i, pusher in enumerate(self.pushers):
            pusher.q = np.hstack([self.c + self.rel_pose[i][:2]@_rot[:2,:2], self.rot + self.rel_pose[i][2]])
            pusher.v = np.hstack([self.v[:2] + np.cross(_w,pusher.q@_rot)[:2], self.v[2]])


    @property
    def r(self):
        return np.hstack([pusher.r for pusher in self.pushers])
    
    @property
    def c(self):
        return self.q[:2]

    @property
    def rot(self):
        return self.q[2]

    @property
    def q_pusher(self):
        return np.hstack([pusher.q for pusher in self.pushers])
    
    @property
    def v_pusher(self):
        return np.hstack([pusher.v for pusher in self.pushers])