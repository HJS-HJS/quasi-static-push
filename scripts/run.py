import time
import numpy as np
import pygame

from utils.object_circle import ObjectCircle
from utils.object_pusher import ObjectPusher
from utils.object_slider import ObjectSlider
from utils.param_function import ParamFunction
from utils.quasi_state_sim import QuasiStateSim

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
pusher_num, pusher_heading = 2, 0
# pusher_num, pusher_heading = 3, np.pi/6
pusher_radius = 0.1
# pusher_distance = 0.25
pusher_distance = 0.4
pusher_position, pusher_rotation = np.array([0.0, -1.0]), 0 #np.array([-0.5, 0.0]), np.pi/2
# object
# slider_radius = np.array([0.5, 0.3])
# slider_position = np.array([[0.0, 0.1], [1.0, -2.0]])
slider_radius = np.array([0.5])
slider_position = np.array([[0.0, 0.1]])

# Set speed 
u_input = np.array([0.0, 0.0, 0.0])
unit_v_speed = 0.5  # [m/s]
unit_r_speed = 0.8  # [rad/s]

# Set simulate param
fps = 60
frame = 1/fps    # 1 frame = 1/fps
sim_step = 100
simulator = QuasiStateSim(sim_step)

# Set pusher and object as object class
pushers = ObjectPusher(pusher_num, pusher_radius, pusher_distance, pusher_heading, pusher_position[0], pusher_position[1], pusher_rotation)

sliders = ObjectSlider()
for i in range(len(slider_radius)):
    sliders.append(ObjectCircle(slider_radius[i], slider_position[i][0], slider_position[i][1]))

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
    if keys[pygame.K_w]:    u_input[0] = unit_v_speed
    elif keys[pygame.K_s]:  u_input[0] = -unit_v_speed
    else:                   u_input[0] = 0
    # Move pusher center in x-axis (ad)
    if keys[pygame.K_a]:    u_input[1] = unit_v_speed
    elif keys[pygame.K_d]:  u_input[1] = -unit_v_speed
    else:                   u_input[1] = 0
    # Rotate pusher center (qe)
    if keys[pygame.K_q]:    u_input[2] = unit_r_speed
    elif keys[pygame.K_e]:  u_input[2] = -unit_r_speed
    else:                   u_input[2] = 0

    # Update pusher center position
    _rot = np.array([
        [-np.sin(pushers.rot), -np.cos(pushers.rot)],
        [np.cos(pushers.rot), -np.sin(pushers.rot)]
        ])

    # run simulator
    qs, qp = simulator.run(
        u_input = np.hstack([_rot@u_input[:2], u_input[2]]) * frame,
        qs      = param.qs,
        qp      = param.qp,
        phi     = param.phi,
        JNS     = param.JNS,
        JNP     = param.JNP,
        JTS     = param.JTS,
        JTP     = param.JTP,
        )

    # Update sliders
    for idx, slider in enumerate(sliders):
        # Update velocity
        slider.v = (qs[idx*3:idx*3 + 3] - slider.q) / frame
        # Update position
        slider.q = qs[idx*3:idx*3 + 3]

    # Update pusher
    # Update velocity
    pushers.apply_v((qp - param.qp) / frame)
    # Update position
    pushers.q = qp

    # Draw Objects
    list(map(lambda pusher: draw_pusher(pusher, unit, display_center, RED), pushers.q_pusher))  # Draw pushers
    list(map(lambda slider: draw_circle(slider, unit, display_center, BLUE), sliders)) # Draw sliders

    # Update display
    pygame.display.update()

    # Set fps
    clock.tick(fps)
    print("Time spent:", time.time() - now)

# Exit simulator
pygame.quit()
