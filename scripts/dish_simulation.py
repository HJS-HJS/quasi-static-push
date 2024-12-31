import os
import time
import random
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

class DishSimulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', random:bool = True, action_skip:int = 5):
        """
        state : image, information
        """
        # Get initial param
        self.state = state
        self.action_skip = action_skip
        self.random = random
        self.gripper_on = False

        # Set pygame display
        if visualize == "human":
            print("[Info] simulator is visulaized")
        elif visualize is None:
            print("[Info] simulator is NOT visulaized")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            print("[Info] simulator is visulaized")

        ## Get config file
        with open("../config/config.yaml") as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # Set patameters
        WIDTH, HEIGHT = self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] # Get pygame display size parameter from config.yaml
        self.display_center = np.array([WIDTH/2, HEIGHT/2])                          # Get center pixel of pygame display

        ## Set parameters
        # Set pixel unit
        self.unit = self.config["display"]["unit"] #[m/pixel]

        # Set pusher
        pusher_num      = self.config["pusher"]["pusher_num"]
        pusher_angle    = np.deg2rad(self.config["pusher"]["pusher_angle"])
        pusher_type     = self.config["pusher"]["pusher_type"]
        pusher_distance = self.config["pusher"]["pusher_distance"]
        pusher_d_u_limit= self.config["pusher"]["pusher_d_u_limit"]
        pusher_d_l_limit= self.config["pusher"]["pusher_d_l_limit"] 

        pusher_position = self.config["pusher"]["pusher_position"]
        pusher_rotation = np.deg2rad(self.config["pusher"]["pusher_rotation"])

        # Set pusher speed
        u_input = np.zeros(3)                            # Initialize pusher's speed set as zeros 
        self.width   = pusher_distance                        # Initialize pusher's speed set as zeros 
        unit_v_speed = self.config["pusher"]["unit_v_speed"]  # [m/s]
        unit_r_speed = self.config["pusher"]["unit_r_speed"]  # [rad/s]
        unit_w_speed = self.config["pusher"]["unit_w_speed"]  # [m/s]
        self.unit_speed = [unit_v_speed, unit_v_speed, unit_r_speed]

        # Set simulate param
        fps = self.config["simulator"]["fps"]           # Get simulator fps from config.yaml
        self.frame = 1 / fps                       # 1 frame = 1/fps
        sim_step = self.config["simulator"]["sim_step"] # Maximun LCP solver step
        self.dist_threshold = float(self.config["simulator"]["dist_threshold"]) # Distance to decide whether to calculate parameters


        ## Generate objects
        # Generate pushers
        self.pushers = ObjectPusher(pusher_num, pusher_angle, pusher_type, pusher_distance, pusher_d_u_limit, pusher_d_l_limit, pusher_position[0], pusher_position[1], pusher_rotation)

        # Generate quasi-static simulation class
        self.simulator = QuasiStateSim(sim_step)

        self.reset()

    def reset(self):
        
        ## Set pygame display settings
        WIDTH, HEIGHT = self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] # Get pygame display size parameter from config.yaml
        # Initialize pygame
        pygame.init()                                       # Initialize pygame
        pygame.display.set_caption("Quasi-static pushing")  # Set pygame display window name
        clock = pygame.time.Clock()                         # Generate pygame clock for apply proper fps
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))   # Set pygame display size
        if not self.random:
            self.table_limit = np.array([WIDTH, HEIGHT])
        else:
            rand_x = random.randint(WIDTH // 2, int(WIDTH * 0.9))
            rand_y = random.randint(HEIGHT // 2, int(HEIGHT * 0.9))
            self.table_limit = np.array([rand_x, rand_y])
        self.backgound = self.create_background_surface(WIDTH, HEIGHT, self.table_limit, self.unit) # Generate pygame background surface
        self.table_limit = self.table_limit*self.unit/2

        ## Generate objects
        # Initialize sliders
        self.pushers.apply_q([0., 0., 0., self.width])
        self.pushers.apply_v([0., 0., 0., 0.])
        if not self.random:
            # Set sliders
            slider_diagram = self.config["sliders"]             # Get sliders property (type, pose, parameters)
            # Set obstacles
            obstacle_diagram = self.config["obstacles"]         # Get sliders property (type, pose, parameters)
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
        else:
            _slider_num = random.randint(1, self.config["auto"]["maximun_number"]) # Get sliders number
            _min_r = self.config["auto"]["minimum_radius"]
            _max_r = self.config["auto"]["maximum_radius"]

            # Generate sliders
            sliders = ObjectSlider()
            _margin  = _min_r
            points, radius = self.generate_spawn_points(_slider_num, _min_r, _max_r, self.table_limit, _margin)
            for point, _r in zip(points, radius):
                a = np.clip(random.uniform(0.8, 1.0) * _r, a_min=_min_r, a_max=_r)
                b = np.clip(random.uniform(0.75, 1.25) * a, a_min=_min_r, a_max=_r)
                r = random.uniform(0, np.pi * 2)
                sliders.append(Ellipse(np.hstack((point,[r])), a, b))
            obstacles = ObjectObstacle()


        # Generate pygame object surfaces
        for pusher in self.pushers: pusher.polygon = self.create_polygon_surface(pusher.torch_points.cpu().numpy().T, COLOR["RED"],   self.unit) # Generate pygame pushers surface
        for slider in sliders:      slider.polygon = self.create_polygon_surface(slider.torch_points.cpu().numpy().T, COLOR["BLUE"],  self.unit) # Generate pygame sliders surface
        for obstac in obstacles:    obstac.polygon = self.create_polygon_surface(obstac.torch_points.cpu().numpy().T, COLOR["GREEN"], self.unit) # Generate pygame sliders surface
        sliders[0].polygon                         = self.create_polygon_surface(sliders[0].torch_points.cpu().numpy().T, COLOR["YELLOW"],  self.unit) # Generate pygame sliders surface

        # Quasi-static simulation class
        # Generate parameter functions
        self.param = ParamFunction(sliders,
                                   self.pushers, 
                                   obstacles, 
                                   self.dist_threshold,
                                   )
        return self.generate_result()

    def step(self, action):
        """
        action: 
        """
        # Limit pusher speed
        if np.abs(action[0]) > self.unit_speed[0]: action[0] = np.sign(action[0]) * self.unit_speed[0] # Limit forward direction speed
        if np.abs(action[1]) > self.unit_speed[0]: action[1] = np.sign(action[1]) * self.unit_speed[0] # Limit sides direction speed
        if np.abs(action[2]) > self.unit_speed[1]: action[2] = np.sign(action[2]) * self.unit_speed[1] # Limit rotation speed

        for _ in range(self.action_skip):
            # Update parameters for quasi-state simulation
            self.param.update_param()
            # Get parameters for simulations
            _qs, _qp, _phi, _JNS, _JNP, _JTS, _JTP, _mu, _A, _B = self.param.get_simulate_param()
            if action[4]:
                if not self.gripper_on:
                    if np.min(_phi[:len(self.param.pushers) * len(self.param.sliders)]) < -0.001: 
                        success = False
                        break
                self.gripper_on = True
                # Generate rotation matrix for pusher velocity input
                _rot = np.array([
                    [-np.sin(self.param.pushers.rot), -np.cos(self.param.pushers.rot)],
                    [np.cos(self.param.pushers.rot),  -np.sin(self.param.pushers.rot)]
                    ])
                # Run quasi-static simulator
                action[:4] += (np.random.random(4) - 0.5) * 0.0005
                qs, qp, success = self.simulator.run(
                    u_input = np.hstack([_rot@action[:2], action[2], action[3]]) * self.frame,
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

                ## Update simulation results
                if success:
                    self.param.sliders.apply_v((qs - _qs) / self.frame) # Update slider velocity
                    self.param.sliders.apply_q(qs)                                # Update slider position
                    self.param.pushers.apply_v((qp - _qp) / self.frame) # Update pusher velocity
                    self.param.pushers.apply_q(qp)                                # Update pusher position
                else: print("\tfailed")
            else:
                _rot = np.array([
                    [-np.sin(self.param.pushers.rot), -np.cos(self.param.pushers.rot)],
                    [np.cos(self.param.pushers.rot),  -np.sin(self.param.pushers.rot)]
                    ])
                self.param.pushers.apply_q(_qp + np.hstack([_rot@action[:2], action[2], action[3]]) * self.frame)
                success = True
                self.gripper_on = False


        # Update pygame display
        # Bliting background
        self.screen.blit(self.backgound, (0, 0))
        # Bliting sliders
        for slider in self.param.sliders:
            _center = slider.q
            _surface = slider.surface([
                int(_center[0]/self.unit + self.display_center[0]), 
                int(-_center[1]/self.unit + self.display_center[1]), 
                _center[2]
                ])
            self.screen.blit(_surface[0], _surface[1])
        # Bliting pushers
        for pusher in self.param.pushers:
            _center = pusher.q
            _surface = pusher.surface([
                int(_center[0]/self.unit + self.display_center[0]), 
                int(-_center[1]/self.unit + self.display_center[1]), 
                _center[2]
                ])
            self.screen.blit(_surface[0], _surface[1])

        # Show updated pygame display
        pygame.display.flip()
        return self.generate_result(success)

    def generate_result(self, success:bool = True):
        """
        state, reward, done
        """
        ## state
        if self.state == 'image':
            # image 
            surface = pygame.display.get_surface()
            state = pygame.surfarray.array3d(surface)
        else:
            state = None

        ## reward
        reward = 0.0
        # time
        reward += -0.5
        # distance
        dist = np.linalg.norm(self.param.sliders[0].q[0:2] - self.param.pushers.q[0:2])
        reward += 5 / np.exp(dist)
        # failed pusher on the slider
        if success: reward += 10
        if not success: reward -= 10

        ## done
        done = False
        for slider in self.param.sliders:
            if np.any(np.abs(slider.q[0:2]) > self.table_limit):
                done = True
                reward += -10
                break
        
        return state, reward, done
    
    def generate_spawn_points(self, num_points, min_r, max_r, limit, margin, center_bias=0.8):
        points = []
        x_range = (-limit[0] + margin, limit[0] - margin)
        y_range = (-limit[1] + margin, limit[1] - margin)

        # 첫 번째 점을 랜덤하게 생성
        center_x = random.uniform(*x_range)
        center_y = random.uniform(*y_range)
        points.append((center_x, center_y))

        # Raduis of inital point
        init_r = random.uniform(min_r, max_r)
        available_lengh = (init_r + min_r, init_r + max_r)
        
        # 나머지 점 생성
        candidate_points = []
        for _ in range(num_points - 1):
            # 첫 번째 점 주변에서 가우시안 분포로 점 생성
            if random.random() < center_bias:  # 중심 근처에 생성될 확률
                new_x = np.clip(np.random.normal(center_x, random.uniform(*available_lengh)), *x_range)
                new_y = np.clip(np.random.normal(center_y, random.uniform(*available_lengh)), *y_range)
            else:  # 전체 영역에 균일 분포로 생성
                new_x = random.uniform(*x_range)
                new_y = random.uniform(*y_range)
            candidate_points.append((new_x, new_y))
        
        # 거리 조건을 만족하는 점만 선택
        for point in candidate_points:
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            if all(d >= (init_r + min_r) for d in distances):
                points.append(point)
        
        points = np.array(points)

        min_distances = np.ones(len(points)) * min_r
        min_distances[0] = init_r

        for idx, point in enumerate(points):
            if idx == 0: continue
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            distances = distances - min_distances
            distances[idx] = max_r
            min_distances[idx] = min(distances)

        # 첫 번째 점을 포함한 최종 점 리스트
        return points, np.array(min_distances)

    def keyboard_input(self, action):
        # Keyboard event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        # Keyboard input
        keys = pygame.key.get_pressed()

        ## Keyboard input response
        # Move pusher center in y-axis (ws)
        if keys[pygame.K_w]:   action[0] +=  self.unit_speed[0]/10  # Move forward      (w)
        elif keys[pygame.K_s]: action[0] += -self.unit_speed[0]/10  # Move backward     (s)
        else:                  action[0]  =  0                # Stop
        # Move pusher center in x-axis (ad)
        if keys[pygame.K_a]:   action[1] +=  self.unit_speed[0]/10  # Move left         (a)
        elif keys[pygame.K_d]: action[1] += -self.unit_speed[0]/10  # Move right        (d)
        else:                  action[1]  =  0                # Stop
        # Rotate pusher center (qe)
        if keys[pygame.K_q]:   action[2] +=  self.unit_speed[1]/10  # Turn ccw          (q)
        elif keys[pygame.K_e]: action[2] += -self.unit_speed[1]/10  # Turn cw           (e)
        else:                  action[2]  =  0                # Stop
        # Control gripper width (left, right)
        if keys[pygame.K_LEFT]:    action[3] = (-self.unit_speed[2]/10)  # Decrease width
        elif keys[pygame.K_RIGHT]: action[3] = ( self.unit_speed[2]/10)  # Increase width
        else:                      action[3] = 0

        if keys[pygame.K_SPACE]:   action[4] = 1  # Gripper On
        else:                      action[4] = 0  # Gripper Off

        # Limit pusher speed
        if np.abs(action[0]) > self.unit_speed[0]: action[0] = np.sign(action[0]) * self.unit_speed[0] # Limit forward direction speed
        if np.abs(action[1]) > self.unit_speed[0]: action[1] = np.sign(action[1]) * self.unit_speed[0] # Limit sides direction speed
        if np.abs(action[2]) > self.unit_speed[1]: action[2] = np.sign(action[2]) * self.unit_speed[1] # Limit rotation speed

        if keys[pygame.K_r]: 
            return np.zeros_like(action), True

        return action, False

    def create_background_surface(self, width, height, table_size, unit, grid:bool = False):
        # Generate pygame surface
        background_surface = pygame.Surface((width, height))    # Generate pygame surface with specific size
        background_surface.fill(COLOR["BLACK"])                          # Fill surface as white

        # Draw white rectangle at the center
        center_x = width // 2
        center_y = height // 2
        table_rect = pygame.Rect(
            center_x - table_size[0] // 2,  # Top-left x
            center_y - table_size[1] // 2,  # Top-left y
            table_size[0],                  # Width
            table_size[1]                   # Height
        )
        pygame.draw.rect(background_surface, COLOR["WHITE"], table_rect)  # Fill the table area with black

        # Draw gridlines
        # 0.2m spacing
        if grid:
            gap = 1 / unit / 5  # Guideline lengh
            for y_idx in range(int(height / gap)): pygame.draw.line(background_surface, COLOR["LIGHTGRAY"], (0, y_idx * gap), (width, y_idx * gap), 2)  # horizontal gridlines
            for x_idx in range(int(width  / gap)): pygame.draw.line(background_surface, COLOR["LIGHTGRAY"], (x_idx * gap, 0), (x_idx * gap, height), 2) # vertical gridlines
            # 1m spacing
            gap = 1 / unit      # Guideline lengh
            for y_idx in range(int(height / gap)): pygame.draw.line(background_surface, COLOR["DARKGRAY"], (0, y_idx * gap), (width, y_idx * gap), 2)   # horizontal gridlines
            for x_idx in range(int(width  / gap)): pygame.draw.line(background_surface, COLOR["DARKGRAY"], (x_idx * gap, 0), (x_idx * gap, height), 2)  # vertical gridlines
        return background_surface

    def create_polygon_surface(self, points, color, unit):
        # Convert polygon points coordinate to pygame display coordinate\
        _points = points.T / unit

        w_l = np.abs(np.min(_points[:,0])) if np.abs(np.min(_points[:,0])) > np.max(_points[:,0]) else np.max(_points[:,0])
        h_l = np.abs(np.min(_points[:,1])) if np.abs(np.min(_points[:,1])) > np.max(_points[:,1]) else np.max(_points[:,1])

        _points[:,0] =  1.0 * _points[:,0] + w_l
        _points[:,1] = -1.0 * _points[:,1] + h_l

        # Set pygame surface size
        width  = int(w_l * 2)
        height = int(h_l * 2)
        # Generate pygame surface
        polygon_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        # Draw
        pygame.draw.polygon(polygon_surface, color, _points.astype(int).tolist())                               # Draw polygon
        pygame.draw.line(polygon_surface, COLOR["LIGHTGRAY"], (width / 4, height / 2), (width * 3 / 4, height / 2), 3)   # Draw horizontal line
        pygame.draw.line(polygon_surface, COLOR["LIGHTGRAY"], (width / 2, height / 4), (width / 2, height * 3 / 4), 3)   # Draw vertical line
        return polygon_surface

if __name__=="__main__":
    run = DishSimulation()
    action = np.zeros(5) # Initialize pusher's speed set as zeros 
    while True:
        action, reset = run.keyboard_input(action)
        state, reward, done = run.step(action=action)
        if reset or done:
            run.reset()