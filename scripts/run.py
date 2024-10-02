import time
import yaml
import numpy as np
import pygame
from sympy import Matrix

from utils.object_circle import ObjectCircle
from utils.object_pusher import ObjectPusher
from utils.object_slider import ObjectSlider
from utils.param_function import ParamFunction
from utils.quasi_state_sim import QuasiStateSim

from utils.color import COLOR

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

def draw_polygon(obj, q, unit, center, color):
    _points = obj.points(q).T / unit
    _points[:,0] = ( 1.0 * _points[:,0] + center[0])
    _points[:,1] = (-1.0 * _points[:,1] + center[1])
    pygame.draw.polygon(screen, color, _points.astype(np.int32).tolist(), 0)

# Initialize pygame
pygame.init()

# Get config file
with open("../config/config.yaml") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

# Set pygame display
WIDTH, HEIGHT = config["display"]["WIDTH"], config["display"]["HEIGHT"]
display_center = np.array([WIDTH/2, HEIGHT/2])
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quasi-static pushing")

# Set color
WHITE       = COLOR["WHITE"]
RED         = COLOR["RED"]
GREEN       = COLOR["GREEN"]
BLUE        = COLOR["BLUE"]
LIGHTGRAY   = COLOR["LIGHTGRAY"]

## Set parameters
# pixel unit
unit = config["display"]["unit"] #[m/pixel]

# pusher
pusher_num      = config["pusher"]["pusher_num"]
pusher_heading  = config["pusher"]["pusher_heading"]
pusher_radius   = config["pusher"]["pusher_radius"]
pusher_distance = config["pusher"]["pusher_distance"]
pusher_position = config["pusher"]["pusher_position"]
pusher_rotation = config["pusher"]["pusher_rotation"]

# object
slider_radius   = config["slider"]["slider_radius"]
slider_position = config["slider"]["slider_position"]

# Set speed 
u_input = np.array([0.0, 0.0, 0.0])
unit_v_speed = config["pusher"]["unit_v_speed"]  # [m/s]
unit_r_speed = config["pusher"]["unit_r_speed"]  # [rad/s]

# Set simulate param
fps = config["simulator"]["fps"]
frame = 1/fps                              # 1 frame = 1/fps
sim_step = config["simulator"]["sim_step"] # Maximun LCP solver step
simulator = QuasiStateSim(sim_step)

# Set pusher and object as object class
pushers = ObjectPusher(pusher_num, pusher_radius, pusher_distance, pusher_heading, pusher_position[0], pusher_position[1], pusher_rotation)

sliders = ObjectSlider()
for i in range(len(slider_radius)):
    _q = Matrix([slider_position[i]])
    _v = Matrix([0, 0, 0])
    sliders.append(ObjectCircle(_q, _v, slider_radius[i], True))

param = ParamFunction(sliders, pushers, False)
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
    if keys[pygame.K_w]:    u_input[0] = unit_v_speed   # Move forward
    elif keys[pygame.K_s]:  u_input[0] = -unit_v_speed  # Move backward
    else:                   u_input[0] = 0              # Stop
    # Move pusher center in x-axis (ad)
    if keys[pygame.K_a]:    u_input[1] = unit_v_speed   # Move left
    elif keys[pygame.K_d]:  u_input[1] = -unit_v_speed  # Move right
    else:                   u_input[1] = 0              # Stop
    # Rotate pusher center (qe)
    if keys[pygame.K_q]:    u_input[2] = unit_r_speed   # Turn ccw
    elif keys[pygame.K_e]:  u_input[2] = -unit_r_speed  # Turn cw
    else:                   u_input[2] = 0              # Stop

    # Update pusher center position
    _rot = np.array([
        [-np.sin(pushers.rot), -np.cos(pushers.rot)],
        [np.cos(pushers.rot),  -np.sin(pushers.rot)]
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
        slider.v = (qs[idx*3:idx*3 + 3] - slider.q) / frame # Update velocity
        slider.q = qs[idx*3:idx*3 + 3]                      # Update position

    # Update pusher
    pushers.apply_v((qp - param.qp) / frame)                # Update velocity
    pushers.q = qp                                          # Update position

    # Draw Objects
    list(map(lambda pusher: draw_polygon(pusher, pushers.q, unit, display_center, RED), pushers)) # Draw pushers
    list(map(lambda slider: draw_polygon(slider, slider.q, unit, display_center, BLUE), sliders)) # Draw sliders
    
    # Update display
    pygame.display.update()

    # Set fps
    clock.tick(fps)

    # Print spent time taken for one iter
    print("Time spent:", time.time() - now)

# Exit simulator
pygame.quit()
