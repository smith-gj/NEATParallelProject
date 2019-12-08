import multiprocessing as mp
import time
import subprocess
import flappy_bird as fb

configFile = 'config-feedforward.txt'



#out = subprocess.run(["flappy_bird.py", 'config-feedforward.txt'])
for i in range(0, 100):
    out = fb.run('config-feedforward.txt')

