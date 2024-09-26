import time
import numpy as np
import pygame

from utils.object_circle import ObjectCircle
from utils.object_pusher import ObjectPusher
from utils.object_slider import ObjectSlider
from utils.param_function import ParamFunction
# from utils.utils import *

def pygame_display_set():
    screen.fill(WHITE)
    gap = 1 / unit
    # horizontal
    for y_idx in range(int(HEIGHT / gap)):
        y_pos = y_idx * gap
        pygame.draw.line(screen, LIGHTGRAY, (0, y_pos), (WIDTH, y_pos), 2)
    # vertical
    for x_idx in range(int(WIDTH / gap)):
        x_pos = x_idx * gap
        pygame.draw.line(screen, LIGHTGRAY, (x_pos, 0), (x_pos, HEIGHT), 2)
def draw_circle(obj, unit, center, color):
    pygame.draw.circle(screen, color, (int(obj.c[0]/unit + center[0]), int(-obj.c[1]/unit + center[1])), obj.r / unit)
def draw_pusher(pusher, unit, center, color):
    pygame.draw.circle(screen, color, (int(pusher[0]/unit + center[0]), int(-pusher[1]/unit + center[1])), pusher_radius / unit)

# Initialize pygame
pygame.init()

# Set pygame display
WIDTH, HEIGHT = 800, 600
display_center = np.array([WIDTH/2, HEIGHT/2])
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quasi-static pushing")

# Set color
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (200, 200, 200)

## Set parameters
# pixel unit
unit = 0.01 #[m/pixel]

# pusher
pusher_num = 2
pusher_heading = 0 #-np.pi/6
pusher_radius = 0.1
pusher_distance = 0.25
pusher_position, pusher_rotation = np.array([0.0, -1.0]), 0 #np.array([-0.5, 0.0]), np.pi/2
# object
# object_radius = np.array([0.5, 0.3])
# object_position = np.array([[0.0, 0.1], [1.0, -2.0]])
object_radius = np.array([0.5])
object_position = np.array([[0.0, 0.1]])

# Set speed 
input_u = [0.0, 0.0, 0.0]
unit_v_speed = 0.5  # [m/s]
unit_r_speed = 0.8  # [rad/s]
# frame = 0.016777    # 1[frame] = 0.016777[s]
frame = 0.033333    # 1[frame] = 0.033333[s]
_rot = np.eye(2)


# Set pusher and object as object class
pushers = ObjectPusher(pusher_num, pusher_radius, pusher_distance, pusher_heading, pusher_position[0], pusher_position[1], pusher_rotation)

sliders = ObjectSlider()
for i in range(len(object_radius)):
    sliders.append(ObjectCircle(object_radius[i], object_position[i][0], object_position[i][1]))

param = ParamFunction(sliders, pushers)
param.update_param()

# Set FPS
clock = pygame.time.Clock()

# Main Loop
running = True
while running:
    now = time.time()

    pygame_display_set()
    
    param.update_param()

    # print('---')
    # print('q:\t', param.q)
    # print('v:\t', param.v)
    # print('phi:\t', param.phi)
    # print('nhat:\t', param.nhat)
    # print('vc:\t', param.vc)
    # print('')

    # if(param.q is not None ):i = 0
    # if(param.v is not None ):i = 0
    # if(param.phi is not None ):i = 0
    # if(param.nhat is not None ):i = 0
    # if(param.vc is not None ):i = 0
    # if(param.vc_jac is not None ):i = 0
        
    # Keyboard event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Keyboard input
    keys = pygame.key.get_pressed()

    # Finish simulator if esc key is pressed
    if keys[pygame.K_ESCAPE]:
        running = False
    
    # Move pusher center (WASD)
    # Move pusher center in y-axis (WS)
    if keys[pygame.K_w]:    input_u[0] = unit_v_speed
    elif keys[pygame.K_s]:  input_u[0] = -unit_v_speed
    else:                   input_u[0] = 0
    # Move pusher center in x-axis (ad)
    if keys[pygame.K_a]:    input_u[1] = unit_v_speed
    elif keys[pygame.K_d]:  input_u[1] = -unit_v_speed
    else:                   input_u[1] = 0
    # Rotate pusher center (qe)
    if keys[pygame.K_q]:    input_u[2] = unit_r_speed
    elif keys[pygame.K_e]:  input_u[2] = -unit_r_speed
    else:                   input_u[2] = 0






    # Calculate pusher real velocity
    pusher_v = input_u

    # Update pusher center position
    _rot = np.array([
        [-np.sin(pushers.rot), -np.cos(pushers.rot)],
        [np.cos(pushers.rot), -np.sin(pushers.rot)]
        ])

    # Update pusher velocity
    pushers.apply_v(pusher_v)
    # Update pusher position
    pushers.move_q(np.hstack([_rot@pushers.v[:2] * frame, pushers.v[2] * frame]))

    
    # Draw Objects
    list(map(lambda pusher: draw_circle(pusher, unit, display_center, RED), pushers))  # Draw pushers
    list(map(lambda slider: draw_circle(slider, unit, display_center, BLUE), sliders)) # Draw sliders

    # Update display
    # pygame.display.flip()
    pygame.display.update()

    # Set fps
    clock.tick(30)
    print("Time spent:", time.time() - now)

# Exit simulator
pygame.quit()
