import time
import yaml
import numpy as np
import pygame

from utils.diagram         import Circle, Ellipse, SuperEllipse, RPolygon, SmoothRPolygon
from utils.object_obstacle import ObjectObstacle
from utils.object_pusher   import ObjectPusher
from utils.object_slider   import ObjectSlider
from utils.param_function  import ParamFunction
from utils.quasi_state_sim import QuasiStateSim

from utils.color import COLOR

def create_background_surface():
    # Generate pygame surface
    background_surface = pygame.Surface((WIDTH, HEIGHT))    # Generate pygame surface with specific size
    background_surface.fill(WHITE)                          # Fill surface as white
    # Draw gridlines
    # 0.2m spacing
    gap = 1 / unit / 5  # Guideline lengh
    for y_idx in range(int(HEIGHT / gap)): pygame.draw.line(background_surface, LIGHTGRAY, (0, y_idx * gap), (WIDTH, y_idx * gap), 2)  # horizontal gridlines
    for x_idx in range(int(WIDTH  / gap)): pygame.draw.line(background_surface, LIGHTGRAY, (x_idx * gap, 0), (x_idx * gap, HEIGHT), 2) # vertical gridlines
    # 1m spacing
    gap = 1 / unit      # Guideline lengh
    for y_idx in range(int(HEIGHT / gap)): pygame.draw.line(background_surface, DARKGRAY, (0, y_idx * gap), (WIDTH, y_idx * gap), 2)   # horizontal gridlines
    for x_idx in range(int(WIDTH  / gap)): pygame.draw.line(background_surface, DARKGRAY, (x_idx * gap, 0), (x_idx * gap, HEIGHT), 2)  # vertical gridlines
    return background_surface

def create_polygon_surface(points, color):
    # Convert polygon points coordinate to pygame display coordinate\
    _points = points.T / unit

    w_l = np.abs(np.min(_points[:,0])) if np.abs(np.min(_points[:,0])) > np.max(_points[:,0]) else np.max(_points[:,0])
    h_l = np.abs(np.min(_points[:,1])) if np.abs(np.min(_points[:,1])) > np.max(_points[:,1]) else np.max(_points[:,1])

    _points[:,0] =  1.0 * _points[:,0] + w_l
    _points[:,1] = -1.0 * _points[:,1] + h_l

    # Set pygame surface size
    width  = w_l * 2
    height = h_l * 2
    # Generate pygame surface
    polygon_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    # Draw
    pygame.draw.polygon(polygon_surface, color, _points.astype(int).tolist())                               # Draw polygon
    pygame.draw.line(polygon_surface, LIGHTGRAY, (width / 4, height / 2), (width * 3 / 4, height / 2), 3)   # Draw horizontal line
    pygame.draw.line(polygon_surface, LIGHTGRAY, (width / 2, height / 4), (width / 2, height * 3 / 4), 3)   # Draw vertical line
    return polygon_surface

def step_time(step:int = 0, msg:str = ""):
    global timer
    global iter_timer
    if step == 0:
        timer = time.time()
        iter_timer = time.time()
    elif step == -1:
        print("Time spent [Hz]: {:.2f}".format(1 /(time.time() - iter_timer)))
    else:
        # print('Step', step, ':\t{:.10f}'.format(time.time() - timer), '\t', msg)
        timer = time.time()

###############################
### Simulation setting code ###
###############################


## Get config file
with open("../config/config.yaml") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

## Set pygame display
# Set patameters
WIDTH, HEIGHT = config["display"]["WIDTH"], config["display"]["HEIGHT"] # Get pygame display size parameter from config.yaml
display_center = np.array([WIDTH/2, HEIGHT/2])                          # Get center pixel of pygame display

## Set colors
WHITE       = COLOR["WHITE"]
BLACK       = COLOR["BLACK"]
RED         = COLOR["RED"]
GREEN       = COLOR["GREEN"]
BLUE        = COLOR["BLUE"]
YELLOW      = COLOR["YELLOW"],
LIGHTGRAY   = COLOR["LIGHTGRAY"]
DARKGRAY    = COLOR["DARKGRAY"]

## Set parameters
# Set pixel unit
unit = config["display"]["unit"] #[m/pixel]

# Set pusher
pusher_num      = config["pusher"]["pusher_num"]
pusher_angle    = np.deg2rad(config["pusher"]["pusher_angle"])
pusher_type     = config["pusher"]["pusher_type"]
pusher_distance = config["pusher"]["pusher_distance"]
pusher_d_u_limit= config["pusher"]["pusher_d_u_limit"]
pusher_d_l_limit= config["pusher"]["pusher_d_l_limit"]

pusher_position = config["pusher"]["pusher_position"]
pusher_rotation = np.deg2rad(config["pusher"]["pusher_rotation"])

# Set pusher speed
u_input = np.zeros(3)                            # Initialize pusher's speed set as zeros 
width   = pusher_distance                        # Initialize pusher's speed set as zeros 
unit_v_speed = config["pusher"]["unit_v_speed"]  # [m/s]
unit_r_speed = config["pusher"]["unit_r_speed"]  # [rad/s]
unit_w_speed = config["pusher"]["unit_w_speed"]  # [m/s]

# Set sliders
slider_diagram = config["sliders"]         # Get sliders property (type, pose, parameters)

# Set obstacles
obstacle_diagram = config["obstacles"]         # Get sliders property (type, pose, parameters)

# Set simulate param
fps = config["simulator"]["fps"]           # Get simulator fps from config.yaml
frame = 1 / fps                            # 1 frame = 1/fps
sim_step = config["simulator"]["sim_step"] # Maximun LCP solver step
dist_threshold = float(config["simulator"]["dist_threshold"]) # Distance to decide whether to calculate parameters

## Generate objects
# Generate pushers
pushers = ObjectPusher(pusher_num, pusher_angle, pusher_type, pusher_distance, pusher_position[0], pusher_position[1], pusher_rotation)
# Generate sliders
sliders = ObjectSlider()
for slider in slider_diagram:
    if   slider["type"] == "circle":       sliders.append(Circle(slider['q'], slider['r']))
    elif slider["type"] == "ellipse":      sliders.append(Ellipse(slider['q'], slider['a'], slider['b']))
    elif slider["type"] == "superellipse": sliders.append(SuperEllipse(slider['q'], slider['a'], slider['b'], slider['n']))
    elif slider["type"] == "rpolygon":     sliders.append(RPolygon(slider['q'], slider['a'], slider['k']))
    elif slider["type"] == "srpolygon":    sliders.append(SmoothRPolygon(slider['q'], slider['a'], slider['k']))
# Generate Obstacles
obstacles = ObjectObstacle()
for obstacle in obstacle_diagram:
    if   obstacle["type"] == "circle":       obstacles.append(Circle(obstacle['q'], obstacle['r']))
    elif obstacle["type"] == "ellipse":      obstacles.append(Ellipse(obstacle['q'], obstacle['a'], obstacle['b']))
    elif obstacle["type"] == "superellipse": obstacles.append(SuperEllipse(obstacle['q'], obstacle['a'], obstacle['b'], obstacle['n']))
    elif obstacle["type"] == "rpolygon":     obstacles.append(RPolygon(obstacle['q'], obstacle['a'], obstacle['k']))
    elif obstacle["type"] == "srpolygon":    obstacles.append(SmoothRPolygon(obstacle['q'], obstacle['a'], obstacle['k']))

## Set pygame display settings
# Initialize pygame
pygame.init()                                       # Initialize pygame
pygame.display.set_caption("Quasi-static pushing")  # Set pygame display window name
screen = pygame.display.set_mode((WIDTH, HEIGHT))   # Set pygame display size
clock = pygame.time.Clock()                         # Generate pygame clock for apply proper fps
backgound = create_background_surface()             # Generate pygame background surface
# Generate pygame object surfaces
for pusher in pushers:   pusher.polygon = create_polygon_surface(pusher.torch_points.cpu().numpy().T, RED)  # Generate pygame pushers surface
for slider in sliders:   slider.polygon = create_polygon_surface(slider.torch_points.cpu().numpy().T, BLUE) # Generate pygame sliders surface
for obstac in obstacles: obstac.polygon = create_polygon_surface(obstac.torch_points.cpu().numpy().T, GREEN) # Generate pygame sliders surface

# Quasi-static simulation class
# Generate parameter functions
param     = ParamFunction(sliders,
                          pushers, 
                          obstacles, 
                          dist_threshold)
# Generate quasi-static simulation class
simulator = QuasiStateSim(sim_step)


###############################
### Simulation running code ###
###############################

# Main Loop
while True:
    # timer for checking spent time in one iteration.
    start_time = time.time()
    step_time(0, "initialize")
    
    #############################
    # Step1. Keyboard interaction

    # Keyboard event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break

    # Keyboard input
    keys = pygame.key.get_pressed()

    ## Keyboard input response
    if keys[pygame.K_ESCAPE]: break                         # Finish simulator (esc)
    # Move pusher center in y-axis (ws)
    if keys[pygame.K_w]:   u_input[0] +=  unit_v_speed/10  # Move forward      (w)
    elif keys[pygame.K_s]: u_input[0] += -unit_v_speed/10  # Move backward     (s)
    else:                  u_input[0] = 0                  # Stop
    # Move pusher center in x-axis (ad)
    if keys[pygame.K_a]:   u_input[1] +=  unit_v_speed/10  # Move left         (a)
    elif keys[pygame.K_d]: u_input[1] += -unit_v_speed/10  # Move right        (d)
    else:                  u_input[1] = 0                  # Stop
    # Rotate pusher center (qe)
    if keys[pygame.K_q]:   u_input[2] +=  unit_r_speed/10  # Turn ccw          (q)
    elif keys[pygame.K_e]: u_input[2] += -unit_r_speed/10  # Turn cw           (e)
    else:                  u_input[2] = 0                  # Stop
    # Control gripper width (left, right)
    if keys[pygame.K_LEFT]:    pushers.width += -unit_w_speed  # Decrease width
    elif keys[pygame.K_RIGHT]: pushers.width +=  unit_w_speed  # Increase width

    # Limit pusher speed
    if np.abs(u_input[0]) > unit_v_speed: u_input[0] = np.sign(u_input[0]) * unit_v_speed # Limit forward direction speed
    if np.abs(u_input[1]) > unit_v_speed: u_input[1] = np.sign(u_input[1]) * unit_v_speed # Limit sides direction speed
    if np.abs(u_input[2]) > unit_r_speed: u_input[2] = np.sign(u_input[2]) * unit_r_speed # Limit rotation speed
    if   pushers.width > pusher_d_u_limit: pushers.width = pusher_d_u_limit               # Limit gripper width
    elif pushers.width < pusher_d_l_limit: pushers.width = pusher_d_l_limit               # Limit gripper width

    # step time
    step_time(1, "Keyboard interaction")

    #############################
    # Step2. Update parameters
    
    # Update parameters for quasi-state simulation
    param.update_param()
    # Get parameters for simulations
    _qs, _qp, _phi, _JNS, _JNP, _JTS, _JTP, _mu, _A, _B = param.get_simulate_param()
    # step time
    step_time(2, 'Update parameters')
    
    #############################
    # Step3. Run simulation
    
    # Generate rotation matrix for pusher velocity input
    _rot = np.array([
        [-np.sin(pushers.rot), -np.cos(pushers.rot)],
        [np.cos(pushers.rot),  -np.sin(pushers.rot)]
        ])
    # Run quasi-static simulator
    qs, qp = simulator.run(
        u_input = np.hstack([_rot@u_input[:2], u_input[2]]) * frame,
        qs  = _qs,
        qp  = _qp,
        phi = _phi,
        JNS = _JNS,
        JNP = _JNP,
        JTS = _JTS,
        JTP = _JTP,
        mu  = _mu,
        A   = _A,
        B   = _B,
        perfect_u_control = False
        )
    # step time
    step_time(3, 'Run simulation')

    #############################
    # Step4. Apply simuation results

    ## Update simulation results
    sliders.apply_v((qs - param.qs) / frame) # Update slider velocity
    sliders.apply_q(qs)                      # Update slider position
    pushers.apply_v((qp - param.qp) / frame) # Update pusher velocity
    pushers.apply_q(qp)                      # Update pusher position

    # step time
    step_time(4, "Apply simuation results")

    #############################
    # Step5. Visualize

    # Update pygame display
    # Bliting background
    screen.blit(backgound, (0, 0))
    # Bliting pushers
    for pusher in pushers:
        _center = pusher.q
        _surface = pusher.surface([
            int(_center[0]/unit + display_center[0]), 
            int(-_center[1]/unit + display_center[1]), 
            _center[2]
            ])
        screen.blit(_surface[0], _surface[1])
    # Bliting sliders
    for slider in sliders:
        _center = slider.q
        _surface = slider.surface([
            int(_center[0]/unit + display_center[0]), 
            int(-_center[1]/unit + display_center[1]), 
            _center[2]
            ])
        screen.blit(_surface[0], _surface[1])
    # Bliting obstacles
    for obs in obstacles:
        _center = obs.q
        _surface = obs.surface([
            int(_center[0]/unit + display_center[0]), 
            int(-_center[1]/unit + display_center[1]), 
            _center[2]
            ])
        screen.blit(_surface[0], _surface[1])

    # Show updated pygame display
    pygame.display.flip()

    # step time
    step_time(5, 'Visualize')


    # Set fps
    clock.tick(fps)
    # step time
    step_time(6, 'simulation tick timer')
    # Print spent time taken for one iter
    step_time(-1, '')

# Exit simulator
pygame.quit()
