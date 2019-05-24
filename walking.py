from itertools import cycle
import sys, random, math

import pygame
from pygame.locals import *

import pymunk
import pymunk.pygame_util

import neat
from numpy.random import randint, choice

import pickle
import os

SCORE = 0

GENERATION = 0
MAX_FITNESS = 0
BEST_GENOME = 0

def get_joint_vertices(body_position, joint_position, angle):
    x, y = joint_position.rotated(angle)/2 + body_position
    return (x, y)



def normalize(vector):
    vector_a = vector[0]
    vector_b = vector[1]
    total = math.sqrt(vector_a**2+vector_b**2)
    return magnitude*vector_a/total, magnitude*vector_b/total

mass_magnitude = 5000

def rotate(x, y, radians, mass, magnitude):
    """Only rotate a point around the origin (0, 0)."""
    xx = (x * math.cos(radians) + y * math.sin(radians))*mass*magnitude*3000
    yy = (-x * math.sin(radians) + y * math.cos(radians))*mass*magnitude*3000

    return (xx, yy)

def joint_clockwise(upper_body, lower_body, upper_segment, lower_segment, magnitude):

    upper_joint_pos = get_joint_vertices(upper_body.position, upper_segment.b, upper_body.angle)
    lower_joint_pos = get_joint_vertices(lower_body.position, lower_segment.a, lower_body.angle)

    upper_normal = rotate(upper_segment.normal[0], upper_segment.normal[1], upper_body.angle, upper_body.mass, magnitude)
    lower_normal = rotate(lower_segment.normal[0], lower_segment.normal[1], lower_body.angle, lower_body.mass, magnitude)

    pymunk.Body.apply_force_at_local_point(upper_body, upper_normal, upper_joint_pos)
    pymunk.Body.apply_force_at_local_point(lower_body, lower_normal, lower_joint_pos)


def joint_anticlockwise(upper_body, lower_body, upper_segment, lower_segment, magnitude):

    upper_joint_pos = get_joint_vertices(upper_body.position, upper_segment.b, upper_body.angle, magnitude)
    lower_joint_pos = get_joint_vertices(lower_body.position, lower_segment.a, lower_body.angle, magnitude)

    upper_normal = rotate(-upper_segment.normal[0], -upper_segment.normal[1], upper_body.angle, upper_body.mass)
    lower_normal = rotate(-lower_segment.normal[0], -lower_segment.normal[1], lower_body.angle, lower_body.mass)

    pymunk.Body.apply_force_at_local_point(upper_body, upper_normal, upper_joint_pos)
    pymunk.Body.apply_force_at_local_point(lower_body, lower_normal, lower_joint_pos)

class human():

    def __init__(self):

        left_thigh = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        left_thigh.position = (100, 175)

        left_thigh_segment = pymunk.Segment(left_thigh, (0, 25), (0, -25), 2)
        left_thigh_segment.density = 1
        left_thigh_segment.friction = 0.46
        left_thigh_segment.filter = pymunk.ShapeFilter(group=0b1)

        left_knee_position = (100, 150)

        left_leg = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        left_leg.position = (100, 125)

        left_leg_segment = pymunk.Segment(left_leg, (0, 25), (0, -25), 2)
        left_leg_segment.density = 1
        left_leg_segment.friction = 0.8
        left_leg_segment.filter = pymunk.ShapeFilter(group=0b1)

        left_knee_joint = pymunk.PivotJoint(left_thigh, left_leg, left_knee_position)
        left_knee_limit = pymunk.RotaryLimitJoint(left_thigh, left_leg, -math.pi / 2, 0)

        left_ankle_position = (100, 100)

        left_foot = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        left_foot.position = (110, 100)

        left_foot_segment = pymunk.Segment(left_foot, (-10, 0), (10, 0), 2)
        left_foot_segment.density = 1
        left_foot_segment.friction = 0.8
        left_foot_segment.filter = pymunk.ShapeFilter(group=0b1)

        left_ankle_joint = pymunk.PivotJoint(left_leg, left_foot, left_ankle_position)
        left_ankle_limit = pymunk.RotaryLimitJoint(left_leg, left_foot, -math.pi/6, math.pi/6)

        # right_leg

        right_thigh = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        right_thigh.position = (100, 175)

        right_thigh_segment = pymunk.Segment(right_thigh, (0, 25), (0, -25), 2)
        right_thigh_segment.density = 1
        right_thigh_segment.friction = 0.46
        right_thigh_segment.filter = pymunk.ShapeFilter(group=0b1)

        right_knee_position = (100, 150)

        right_leg = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        right_leg.position = (100, 125)

        right_leg_segment = pymunk.Segment(right_leg, (0, 25), (0, -25), 2)
        right_leg_segment.density = 1
        right_leg_segment.friction = 0.8
        right_leg_segment.filter = pymunk.ShapeFilter(group=0b1)

        right_knee_joint = pymunk.PivotJoint(right_thigh, right_leg, right_knee_position)
        right_knee_limit = pymunk.RotaryLimitJoint(right_thigh, right_leg, -math.pi / 2, 0)

        right_ankle_position = (100, 100)

        right_foot = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        right_foot.position = (110, 100)

        right_foot_segment = pymunk.Segment(right_foot, (-10, 0), (10, 0), 2)
        right_foot_segment.density = 1
        right_foot_segment.friction = 0.8
        right_foot_segment.filter = pymunk.ShapeFilter(group=0b1)

        right_ankle_joint = pymunk.PivotJoint(right_leg, right_foot, right_ankle_position)
        right_ankle_limit = pymunk.RotaryLimitJoint(right_leg, right_foot, -math.pi /6, math.pi / 6)

        hip_position = (100, 200)

        torso = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        torso.position = (100, 240)

        torso_segment = pymunk.Segment(torso, (0, 40), (0, -40), 2)
        torso_segment.density = 1
        torso_segment.friction = 0.46
        torso_segment.filter = pymunk.ShapeFilter(group=0b1)

        left_hip_joint = pymunk.PivotJoint(torso, left_thigh, hip_position)
        left_hip_limit = pymunk.RotaryLimitJoint(torso, left_thigh, -math.pi / 2, math.pi / 2)

        right_hip_joint = pymunk.PivotJoint(torso, right_thigh, hip_position)
        right_hip_limit = pymunk.RotaryLimitJoint(torso, right_thigh, -math.pi / 2, math.pi / 2)

        space.add(left_thigh, left_thigh_segment, left_leg, left_leg_segment, left_foot, left_foot_segment)
        space.add(right_thigh, right_thigh_segment, right_leg, right_leg_segment, right_foot, right_foot_segment)

        space.add(torso, torso_segment, left_hip_joint, right_hip_joint, left_knee_joint, right_knee_joint, left_ankle_joint, right_ankle_joint)
        space.add(left_hip_limit, right_hip_limit, left_knee_limit, right_knee_limit, left_ankle_limit, right_ankle_limit)

        self.l_thigh = left_thigh
        self.l_thigh_seg = left_thigh_segment

        self.l_leg = left_leg
        self.l_leg_seg = left_leg_segment

        self.l_foot = left_foot
        self.l_foot_seg = left_foot_segment

        self.r_thigh = right_thigh
        self.r_thigh_seg = right_thigh_segment

        self.r_leg = right_leg
        self.r_leg_seg = right_leg_segment

        self.r_foot = right_foot
        self.r_foot_seg = right_foot_segment

        self.chest = torso
        self.chest_seg = torso_segment

        self.alive = True

    def update(self):
        torso, torso_seg = self.chest, self.chest_seg
        if torso.position[1]-abs(40*math.cos(torso.angle)) < 15:
            self.alive = False

    def move(self, offset):
        body_part = [self.chest, self.l_thigh, self.r_thigh, self.l_leg, self.r_leg, self.l_foot, self.r_foot]
        moved_body_part = []
        for x in body_part:
            pos_x, pos_y = x.position[0], x.position[1]
            x.position = pos_x-offset, pos_y
            moved_body_part.append(x)
        [self.chest, self.l_thigh, self.r_thigh, self.l_leg, self.r_leg, self.l_foot, self.r_foot] = moved_body_part

    def score(self):
        distance = self.chest.position[0]
        return distance

    def levitate(self):
        torso, left_thigh, right_thigh, left_leg, right_leg, left_foot, right_foot = self.chest, self.l_thigh, self.r_thigh, self.l_leg, self.r_leg, self.l_foot, self.r_foot
        if  (   left_thigh.position[1] - abs(25*math.cos(left_thigh.angle)) > 18
            and right_thigh.position[1] - abs(25*math.cos(right_thigh.angle)) > 18
            and left_leg.position[1] - abs(25*math.cos(left_leg.angle)) > 18
            and right_leg.position[1] - abs(25*math.cos(right_leg.angle)) > 18
            and left_foot.position[1] - abs(10*math.cos(left_foot.angle)) > 18
            and right_foot.position[1] - abs(10*math.cos(right_foot.angle)) > 18
            and torso.position[1] - abs(40*math.cos(torso.angle)) > 18):
            return True
        else:
            return False

    #hip


    def left_hip(self, magnitude):
        torso, left_thigh, torso_segment, left_thigh_segment = self.chest, self.l_thigh, self.chest_seg, self.l_thigh_seg

        if abs(torso.angle - left_thigh.angle) < math.pi / 4 or self.levitate():
            return

        joint_clockwise(torso, left_thigh, torso_segment, left_thigh_segment, magnitude)

    def right_hip(self, magnitude):
        torso, right_thigh, torso_segment, right_thigh_segment = self.chest, self.r_thigh, self.chest_seg, self.r_thigh_seg

        if abs(torso.angle - right_thigh.angle) < math.pi / 4 or self.levitate():
            return
        joint_clockwise(torso, right_thigh, torso_segment, right_thigh_segment, magnitude)

    #knee

    def left_knee(self, magnitude):
        left_thigh, left_leg, left_thigh_segment, left_leg_segment = self.l_thigh, self.l_leg, self.l_thigh_seg, self.l_leg_seg
        if left_leg.position[1] - 25*abs(math.sin(left_leg.angle)) > 28:
            return
        joint_clockwise(left_thigh, left_leg, left_thigh_segment, left_leg_segment, magnitude)

    def right_knee(self, magnitude):
        right_thigh, right_leg, right_thigh_segment, right_leg_segment = self.r_thigh, self.r_leg, self.r_thigh_seg, self.r_leg_seg

        if right_leg.position[1] - 25 * abs(math.sin(right_leg.angle)) > 28:
            return

        joint_clockwise(right_thigh, right_leg, right_thigh_segment, right_leg_segment, magnitude)

    #ankle

    def left_ankle(self, magnitude):
        left_leg, left_foot, left_leg_segment, left_foot_segment = self.l_leg, self.l_foot, self.l_leg_seg, self.l_foot_seg
        if left_foot.position[1] - 10*abs(math.sin(left_foot.angle)) > 28:
            return
        joint_clockwise(left_leg, left_foot, left_leg_segment, left_foot_segment, magnitude)

    def right_ankle(self, magnitude):
        right_leg, right_foot, right_leg_segment, right_foot_segment = self.r_leg, self.r_foot, self.r_leg_seg, self.r_foot_seg

        if right_foot.position[1] - 10 * abs(math.sin(right_foot.angle)) > 28:
            return

        joint_clockwise(right_leg, right_foot, right_leg_segment, right_foot_segment, magnitude)

def add_ground(space):

    body = pymunk.Body(0, 10000, body_type = pymunk.Body.STATIC)
    body.position = (0, 0)

    ground = pymunk.Segment(body, (-600, 0), (100000000.0, 0), 5.0)
    ground.filter = pymunk.ShapeFilter(group=0b0)
    ground.friction = 1

    space.add(ground, body)
    return ground

def game(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    global space

    timer = 0

    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Falling L")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, -1800.0)
    space.damping = 0.6

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    ground_object = add_ground(space)

    key_down = []

    sec = 0

    shift = 0

    person = human()

    total_x = 0

    previous_x = 0

    pause = False

    while person.alive:

        input = (person.chest.position[0], person.chest.position[1], person.chest.angle, person.chest.angular_velocity, person.l_thigh.angle, person.l_thigh.angular_velocity, person.r_thigh.angle, person.l_thigh.angular_velocity, person.l_leg.angle, person.l_leg.angular_velocity, person.r_leg.angle, person.r_leg.angular_velocity, person.l_foot.angle, person.l_foot.angular_velocity, person.r_foot.angle, person.r_foot.angular_velocity, person.levitate())

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == pygame.K_s:
                person.alive = False
            elif event.type == KEYDOWN and event.key == pygame.K_r:
                return 0, 1
            elif event.type == KEYDOWN and event.key == pygame.K_b:
                return 0, 2
            elif event.type == KEYDOWN and event.key == pygame.K_SPACE:
                pause = True
                while pause:
                    for event in pygame.event.get():
                        if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                            pause = False
                        elif event.type == KEYDOWN and event.key == pygame.K_r:
                            return 0, 1
                        elif event.type == KEYDOWN and event.key == pygame.K_b:
                            return 0, 2


        output = net.activate(input)

        if person.alive == True:
            person.left_hip(output[0])
            person.right_hip(output[1])
            person.left_knee(output[2])
            person.right_knee(output[3])
            person.left_ankle(output[4])
            person.right_ankle(output[5])

            person.update()

            if person.chest.position[0] > 600:
                total_x += 600
                person.move(600)

            elif person.chest.position[0] < 0:
                total_x -= 600
                person.move(-600)

        screen.fill((255,255,255))

        space.debug_draw(draw_options)

        space.step(1/200.0)

        pygame.display.flip()
        clock.tick(200)
        timer += 1


        if timer%1000 == 0:
            current_x = total_x + person.chest.position[0] - 100
            print(f'total_x = {current_x}')
            if abs(current_x - previous_x) < 10:
                person.alive = False
            previous_x = current_x

    fitness = total_x + person.chest.position[0] - 100 - 20000/timer

    return fitness, 0


def eval_genomes(genomes, config):
    i = 0
    global SCORE
    global GENERATION, MAX_FITNESS, BEST_GENOME, best_genome_list

    best_genome_list = []

    GENERATION += 1

    iteration = len(genomes)
    loop_number = 0

    move_on, store = 1, 0

    fitness_space = 0


    while loop_number < iteration:

        genome_id, genome = genomes[loop_number][0], genomes[loop_number][1]

        while move_on > 0:

            if move_on == 1:
                fitness_space, move_on = game(genome, config)

            if move_on == 2:
                fitness_space, move_on = game(previous_genome, config)
                store = 1

            previous_genome = genome

        if store == 0:
            genome.fitness = fitness_space
            print("Gen : %d Genome # : %d  Fitness : %f Max Fitness : %f" % (GENERATION, i, genome.fitness, MAX_FITNESS))
            if genome.fitness >= MAX_FITNESS:
                MAX_FITNESS = genome.fitness
                BEST_GENOME = genome
                best_genome_list.append(genome)
            i += 1
            loop_number += 1
        else:
            store = 0

        move_on = 1



        SCORE = 0


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.txt') #Make sure config.txt is in the same file as this script

pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(eval_genomes, 40)

print(winner)

print(best_genome_list)

outputDir = '/Users/Godwyn Lai/PycharmProjects/Basketball/venv' #Add where you want to output the resulting file about the best genome to
os.chdir(outputDir)
serialNo = len(os.listdir(outputDir)) + 1
outputFile = open(str(serialNo) + '_' + str(int(MAX_FITNESS)) + '.p', 'wb')

pickle.dump(winner, outputFile)

iterations = range(len(best_genome_list))

while True:
    for x in iterations:
        print(f'best of generation {x}')
        unused_space = game(best_genome_list[x], config)
