import multiprocessing as mp
import time
import subprocess
import flappy_bird as fb

configFile = 'config-feedforward.txt'


def run(sema):
    sema.acquire()
    fb.run(configFile)
    sema.release()


#out = subprocess.run(["flappy_bird.py", 'config-feedforward.txt'])
out = subprocess.Popen('flappy_bird.py')

print("done\n")

