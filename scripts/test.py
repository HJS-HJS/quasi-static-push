import numpy as np

from utils.object_circle import ObjectCircle
from utils.utils import *

# set object property

circle = ObjectCircle(0.5, 0.0, 0.0)
pusher_l = ObjectCircle(0.1, -0.5, 0.01)
pusher_r = ObjectCircle(0.1, -0.5, -0.01)

norm_l = unit_vector(pusher_l.c - circle.c)
norm_r = unit_vector(pusher_r.c - circle.c)

speed_r = circle.point_velocity(norm_r)
speed_l = circle.point_velocity(norm_l)

print(circle.c)
print(pusher_l.c)
print(pusher_r.c)

print(norm_l)
print(norm_r)

print(speed_r)
print(speed_l)

