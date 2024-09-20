import pygame
import numpy as np

from utils.object_circle import ObjectCircle
from utils.object_pusher import ObjectPusher
from utils.utils import *


def draw_circle(obj, unit, center, color):
    pygame.draw.circle(screen, color, (int(obj.c[0]/unit + center[0]), int(-obj.c[1]/unit + center[1])), obj.r / unit)
def draw_pusher(pusher, unit, center, color):
    pygame.draw.circle(screen, color, (int(pusher[0]/unit + center[0]), int(-pusher[1]/unit + center[1])), pusher_radius / unit)
def pygame_display_set():
    screen.fill(WHITE)
    gap = 1 / unit
    # 가로선
    for y_idx in range(int(HEIGHT / gap)):
        y_pos = y_idx * gap
        pygame.draw.line(screen, LIGHTGRAY, (0, y_pos), (WIDTH, y_pos), 2)

    # 세로선
    for x_idx in range(int(WIDTH / gap)):
        x_pos = x_idx * gap
        pygame.draw.line(screen, LIGHTGRAY, (x_pos, 0), (x_pos, HEIGHT), 2)


# Initialize pygame
pygame.init()

# Set pygame display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quasi-static pushing")

# Set color
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (200, 200, 200)

# Set parameters
unit = 0.01
view_center = np.array([WIDTH/2, HEIGHT/2])
# pusher
pusher_num = 2
pusher_radius = 0.1
pusher_distance = 0.25
pusher_position, pusher_rotation = np.array([0.0, -1.0]), 0
# pusher_position, pusher_rotation = np.array([-0.5, 0.0]), np.pi/2
# object
object_num = 1
object_radius = 0.5
object_position = np.array([0.0, 0.1])

# Set pusher and object as object class
pushers = ObjectPusher(pusher_num, pusher_radius, pusher_distance, pusher_position[0], pusher_position[1], pusher_rotation)
objects = []
for i in range(object_num):
    objects.append(ObjectCircle(object_radius, object_position[0], object_position[1]))

# Set speed 
input_u = [0.0, 0.0, 0.0]
unit_v_speed = 0.5  # [m/s]
unit_r_speed = 0.8  # [rad/s]
frame = 0.016777    # 1[frame] = 0.016777[s]
_rot = np.eye(2)

# FPS 설정
clock = pygame.time.Clock()

# 메인 루프
running = True
while running:
    pygame_display_set()

    print('')
    ############# q #############
    # qo
    for idx, object in enumerate(objects):
        print('qo_{}:\t'.format(idx), object.q_deg)
        object.set_v([0, 0, 1])
    # qmg
    print('qm_g:\t',pushers.q_deg)
    # qm
    for idx, pusher_c in enumerate(pushers.finger_c):
        print('qm_{}:\t'.format(idx), pusher_c)
    
    ############# v #############
    # vo
    for idx, object in enumerate(objects):
        print('vo_{}:\t'.format(idx),object.v_deg)
    # vmg
    print('vm_g:\t',pushers.v_deg)
    # vm
    for idx, pusher_v in enumerate(pushers.finger_v):
        print('vm_{}:\t'.format(idx), pusher_v)

    ############# phi #############
    for idx_o, object in enumerate(objects):
        for idx_f, pusher_c in enumerate(pushers.finger_c):
            print('phi_{}:\t'.format(idx_o * pusher_num + idx_f), np.linalg.norm(pusher_c[:2] - object.c) - pushers.r - object.r)

    ############# nhat #############
    ############# vc #############
    unit_vectors = []
    vc_set = []
    for object in objects:
        for finger_c, finger_v in zip(pushers.finger_c, pushers.finger_v):
            unit_vectors.append(unit_vector(finger_c[:2] - object.c))
            vc_set.append(finger_v[:2] - object.point_velocity(unit_vector(finger_c[:2] - object.c)))

    for idx, unit_v in enumerate(unit_vectors):
        print('nhat_{}:\t'.format(idx), unit_v)

    for idx, vc in enumerate(vc_set):
        print('vc_{}:\t'.format(idx), vc)


    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 키 입력 처리
    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        running = False
    
    # 중심 이동 (WASD)
    if keys[pygame.K_w]:
        input_u[0] = unit_v_speed
    elif keys[pygame.K_s]:
        input_u[0] = -unit_v_speed
    else:
        input_u[0] = 0

    if keys[pygame.K_a]:
        input_u[1] = unit_v_speed
    elif keys[pygame.K_d]:
        input_u[1] = -unit_v_speed
    else:
        input_u[1] = 0

    # 회전 처리 (Q, E)
    if keys[pygame.K_q]:
        input_u[2] = unit_r_speed
    elif keys[pygame.K_e]:
        input_u[2] = -unit_r_speed
    else:
        input_u[2] = 0


    # 중심 위치 업데이트
    _rot = np.array([
        [-np.sin(pushers.rot), -np.cos(pushers.rot)],
        [np.cos(pushers.rot), -np.sin(pushers.rot)]
        ])

    pusher_v = input_u

    pushers.set_v(pusher_v)
    pushers.rotate_pusher(pushers.v[2] * frame)
    pushers.move_pusher(_rot@pushers.v[:2] * frame)

    # 원 그리기 (빨간색과 초록색 원)
    list(map(lambda pusher: draw_pusher(pusher, unit, view_center, LIGHTGRAY), pushers.finger_c))

    # 고정된 파란색 원 (중심 고정)
    for object in objects:
        draw_circle(object, unit, view_center, BLUE)

    # 화면 업데이트
    pygame.display.flip()

    # FPS 설정
    clock.tick(60)
# 게임 종료
pygame.quit()


