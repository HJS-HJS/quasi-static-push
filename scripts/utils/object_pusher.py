import numpy as np
from utils.object_circle import ObjectCircle

class ObjectPusher(object):
    '''
    2d pusher
    '''
    def __init__(self, n_finger, radius, distance, heading, center_x, center_y, rotation):
        self.fingers = []
        for i in range(n_finger):
            self.fingers.append(ObjectCircle(radius,
                                             distance * np.cos(2 * np.pi / n_finger * i + heading),
                                             distance * np.sin(2 * np.pi / n_finger * i + heading),
                                             2 * np.pi / n_finger * i
                                             ))

        self.radius = radius
        self.q = np.array([center_x, center_y, rotation])
        self.velocity = np.array([0, 0, 0])

    def point_velocity(self, norm):
        _arr = np.array([[0, -1], [1, 0]])
        return self.velocity[2] * self.radius * _arr @ norm + self.velocity[:2]

    def set_c(self, center):
        self.q[:2] = center
        
    def set_v(self, velocity):
        self.velocity = velocity

    def move_pusher(self, d_center):
        self.q[:2] += d_center
    
    def rotate_pusher(self, r_center):
        self.q[2] += r_center
    
    
    @property
    def finger_r(self):
        return np.array([finger.r for finger in self.fingers])
    
    @property
    def finger_c(self):
        _rot = np.array([
            [np.cos(self.q[2]), np.sin(self.q[2])],
            [np.sin(self.q[2]), -np.cos(self.q[2])],
            ])
        return np.array([np.hstack([self.c + finger.c@_rot, self.rot + finger.rot]) for finger in self.fingers])
        
    @property
    def finger_v(self):
        _rot = np.eye(3)
        _rot[:2,:2] = np.array([
            [np.cos(self.q[2]), np.sin(self.q[2])],
            [np.sin(self.q[2]), -np.cos(self.q[2])],
            ])
        _w = np.array([0, 0, self.v[2]])
        return np.array([np.hstack([self.v[:2] + np.cross(_w,finger.q@_rot)[:2], self.v[2]]) for finger in self.fingers])

    @property
    def r(self):
        return self.radius
    
    @property
    def q_deg(self):
        return np.hstack([self.c, np.rad2deg(self.rot)])
    
    @property
    def c(self):
        return self.q[:2]
        
    @property
    def v(self):
        return self.velocity
    
    @property
    def v_deg(self):
        return np.hstack([self.velocity[:2], np.rad2deg(self.velocity[2])])
    
    @property
    def rot(self):
        return self.q[2]
