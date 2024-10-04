import time
import yaml
import numpy as np
import pygame
from sympy import Matrix, MatrixSymbol

from utils.diagram         import Diagram, Circle
from utils.object_pusher   import ObjectPusher
from utils.object_slider   import ObjectSlider
from utils.param_function  import ParamFunction
from utils.quasi_state_sim import QuasiStateSim

from utils.color import COLOR

def create_background_surface():
    # Generate pygame surface
    background_surface = pygame.Surface((WIDTH, HEIGHT))
    # Fill surface as white
    background_surface.fill(WHITE)
    # Draw gridlines
    gap = 1 / unit
    # horizontal gridlines
    for y_idx in range(int(HEIGHT / gap)): pygame.draw.line(background_surface, LIGHTGRAY, (0, y_idx * gap), (WIDTH, y_idx * gap), 2)
    # vertical gridlines
    for x_idx in range(int(WIDTH / gap)): pygame.draw.line(background_surface, LIGHTGRAY, (x_idx * gap, 0), (x_idx * gap, HEIGHT), 2)
    return background_surface

def create_polygon_surface(points, color):
    # Convert polygon points coordinate to pygame display coordinate
    _points = points.T / unit
    _points[:,0] = ( 1.0 * _points[:,0] + display_center[0])
    _points[:,1] = (-1.0 * _points[:,1] + display_center[1])
    _points[:,0] -= np.min(_points[:,0])
    _points[:,1] -= np.min(_points[:,1])
    # Set pygame surface size
    width = np.max([p[0] for p in _points]) - np.min([p[0] for p in _points])
    height = np.max([p[1] for p in _points]) - np.min([p[1] for p in _points])
    # Generate pygame surface
    polygon_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    # Draw
    pygame.draw.polygon(polygon_surface, color, _points.astype(int).tolist())                               # Draw polygon
    pygame.draw.line(polygon_surface, LIGHTGRAY, (width / 4, height / 2), (width * 3 / 4, height / 2), 1)   # Draw horizontal line
    pygame.draw.line(polygon_surface, LIGHTGRAY, (width / 2, height / 4), (width / 2, height * 3 / 4), 1)   # Draw vertical line

    return polygon_surface

### Simulation running code

## Get config file
with open("../config/config.yaml") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

## Set pygame display
# Set patameters
WIDTH, HEIGHT = config["display"]["WIDTH"], config["display"]["HEIGHT"] # Get pygame display size parameter from config.yaml
display_center = np.array([WIDTH/2, HEIGHT/2])                          # Get center pixel of pygame display

## Set colors
WHITE       = COLOR["WHITE"]
RED         = COLOR["RED"]
GREEN       = COLOR["GREEN"]
BLUE        = COLOR["BLUE"]
LIGHTGRAY   = COLOR["LIGHTGRAY"]

## Set parameters
# Set pixel unit
unit = config["display"]["unit"] #[m/pixel]

# Set pusher
pusher_num      = config["pusher"]["pusher_num"]
pusher_heading  = config["pusher"]["pusher_heading"]
pusher_radius   = config["pusher"]["pusher_radius"]
pusher_distance = config["pusher"]["pusher_distance"]
pusher_position = config["pusher"]["pusher_position"]
pusher_rotation = np.deg2rad(config["pusher"]["pusher_rotation"])

# Set pusher speed
u_input = np.zeros(3)                            # Initialize pusher's speed set as zeros 
unit_v_speed = config["pusher"]["unit_v_speed"]  # [m/s]
unit_r_speed = config["pusher"]["unit_r_speed"]  # [rad/s]

# Set objects
slider_radius   = config["slider"]["slider_radius"]
slider_position = config["slider"]["slider_position"]

# Set simulate param
fps = config["simulator"]["fps"]           # Get simulator fps from config.yaml
frame = 1 / fps                            # 1 frame = 1/fps
sim_step = config["simulator"]["sim_step"] # Maximun LCP solver step

## Generate objects
# Generate pushers
pushers = ObjectPusher(pusher_num, pusher_radius, pusher_distance, pusher_heading, pusher_position[0], pusher_position[1], pusher_rotation)
# Generate sliders
sliders = ObjectSlider()
for i in range(len(slider_radius)):
    _q = np.array(slider_position[i])
    _v = np.array([0, 0, 0])
    sliders.append(Circle(_q, _v, slider_radius[i], None))

## Set pygame display settings
pygame.init()                                       # Initialize pygame
pygame.display.set_caption("Quasi-static pushing")  # Set pygame display window name
screen = pygame.display.set_mode((WIDTH, HEIGHT))   # Set pygame display size
clock = pygame.time.Clock()                         # Generate pygame clock for apply proper fps
backgound = create_background_surface()             # Generate pygame background surface
# Generate pygame pushers surface
for pusher in pushers:
    _center = np.array(pusher.q.subs({MatrixSymbol('qp_', 1, 3):Matrix(pushers.q.reshape(1,3))})).astype(float)[0]
    pusher.polygon = create_polygon_surface(pusher.points(pushers.q), RED)
    pusher.init_angle = _center[2]
# Generate pygame sliders surface
for slider in sliders:
    _center = slider.q
    slider.polygon = create_polygon_surface(slider.points(slider.q), BLUE)
    slider.init_angle = _center[2]

# Quasi-static simulation class
param = ParamFunction(sliders, pushers, False)  # Generate parameter functions
simulator = QuasiStateSim(sim_step)             # Generate quasi-static simulation class

# Main Loop
running = True
while running:
    now = time.time()
    now1 = time.time()

    # Update parameters for quasi-state simulation
    param.update_param()

    # Keyboard event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Keyboard input
    keys = pygame.key.get_pressed()

    ## Keyboard input response
    if keys[pygame.K_ESCAPE]: running = False           # Finish simulator (esc)
    # Move pusher center in y-axis (ws)
    if keys[pygame.K_w]:    u_input[0] = unit_v_speed   # Move forward      (w)
    elif keys[pygame.K_s]:  u_input[0] = -unit_v_speed  # Move backward     (s)
    else:                   u_input[0] = 0              # Stop
    # Move pusher center in x-axis (ad)
    if keys[pygame.K_a]:    u_input[1] = unit_v_speed   # Move left         (a)
    elif keys[pygame.K_d]:  u_input[1] = -unit_v_speed  # Move right        (d)
    else:                   u_input[1] = 0              # Stop
    # Rotate pusher center (qe)
    if keys[pygame.K_q]:    u_input[2] = unit_r_speed   # Turn ccw          (q)
    elif keys[pygame.K_e]:  u_input[2] = -unit_r_speed  # Turn cw           (e)
    else:                   u_input[2] = 0              # Stop

    print('\tstep1:\t{:.10f}'.format(time.time() - now1))
    now1 = time.time()

    # Generate rotation matrix for pusher velocity input
    _rot = np.array([
        [-np.sin(pushers.rot), -np.cos(pushers.rot)],
        [np.cos(pushers.rot),  -np.sin(pushers.rot)]
        ])
    
    # Run quasi-static simulator
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

    print('\tstep2:\t{:.10f}'.format(time.time() - now1))
    now1 = time.time()

    ## Update simulation results
    sliders.apply_v((qs - param.qs) / frame)       # Update slider velocity
    sliders.apply_q(qs)                            # Update slider position
    pushers.apply_v((qp - param.qp) / frame)       # Update pusher velocity
    pushers.apply_q(qp)                            # Update pusher position

    print('\tstep3:\t{:.10f}'.format(time.time() - now1))
    now1 = time.time()

    # Update pygame display
    # Bliting background
    screen.blit(backgound, (0, 0))
    # Bliting pushers
    for pusher in pushers:
        _center = np.array(pusher.q.subs({MatrixSymbol('qp_', 1, 3):Matrix(pushers.q.reshape(1,3))})).astype(float)[0]
        _surface = pusher.surface([int(_center[0]/unit + display_center[0]), int(-_center[1]/unit + display_center[1]), _center[2]])
        screen.blit(_surface[0], _surface[1])
    # Bliting sliders
    for slider in sliders:
        _center = slider.q
        _surface = slider.surface([int(_center[0]/unit + display_center[0]), int(-_center[1]/unit + display_center[1]), _center[2]])
        screen.blit(_surface[0], _surface[1])

    # Show updated pygame display
    pygame.display.flip()

    print('\tstep4:\t{:.10f}'.format(time.time() - now1))
    now1 = time.time()

    # Set fps
    clock.tick(fps)

    # Print spent time taken for one iter
    print('\tstep5:\t{:.10f}'.format(time.time() - now1))
    print("Time spent:\t{:.10f}".format(time.time() - now))

# Exit simulator
pygame.quit()
