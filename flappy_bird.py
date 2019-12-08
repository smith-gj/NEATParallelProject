"""
CS 6068 Final Project

A parallelized implementation of a NEAT algorithm playing Flappy Bird
    Nick Heminger
    Thomas Kissel
    Gabe Smith

Flappy Bird game logic credit to:
    https://github.com/techwithtim/NEAT-Flappy-Bird
"""
import random
import os
import time
import neat
import visualize
import pickle
import numpy as np
import multiprocessing as mp

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
DRAW_LINES = False

gen = 0

class Bird:
    """
    Bird class representing the flappy bird
    """
    MAX_ROTATION = 25
    # IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 8
    INDEX = 0

    def __init__(self, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        # self.img = self.IMGS[0]

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL


class Pipe():
    """
    represents a pipe object
    """
    GAP = 200
    VEL = 8

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        # self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        # self.PIPE_BOTTOM = pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(50, 450)
        self.top = self.height # - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def collide(self, bird):
        if bird.y >= self.height or bird.y <= self.height:
            return True

        return False


class Base:
    """
    Represnts the moving floor of the game
    """
    VEL = 8
    WIDTH = 10 #base_img.get_width()
    # IMG = base_img

    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH


def decide_birds(nets, birds, bird, pipe_top, pipe_bot):
    bird.move()
    # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
    output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipe_top), abs(bird.y - pipe_bot)))
    if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
        bird.jump()


def call_activate(holder):
    for entry in holder:
        index = entry[2]
        nets = entry[3]
        output = nets[index].activate(entry[0].y, entry[4], entry[5])
        if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
            entry[0].jump()


def decide_birds_parallel(nets, birds, pipe):
    holder = []
    for bird in birds:
        bird.move()
        index = birds.index(bird)
        holder.append([bird, index, nets, abs(bird.y - pipe.top), abs(bird.y - pipe.bottom)])

    map(call_activate, holder)


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    global gen
    #win = WIN
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        index = 0
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        birds[index].index = index
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    run = True
    while run and len(birds) > 0:

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x:  # determine whether to use the first or second
                pipe_ind = 1

        top_pipe = pipes[pipe_ind].height
        bot_pipe = pipes[pipe_ind].bottom
        pipe = pipes[pipe_ind]

        #decide_birds_parallel(nets, birds, pipe)

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            decide_birds(nets, birds, bird, top_pipe, bot_pipe)

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            for bird in birds:
                if pipe.collide(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        # break if score gets large enough
        if score > 20:
            pickle.dump(nets[0], open("best.pickle", "wb"))
            break


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    time_start = time.time()
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(True))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    #print('\nBest genome:\n{!s}'.format(winner))
    time_end = time.time() - time_start
    print(time_end)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
