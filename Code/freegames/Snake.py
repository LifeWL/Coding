from random import randrange
from turtle import *

from freegames import square, vector

food = vector(0, 0)
snake = [vector(10, 0)]
aim = vector(0, -10)

def change(x, y):
    aim.x = X
    aim.y = y

def inside(head):
    return -200 < head.x < 190 and -200 < head.y < 190

def move():
    head = snake[-1].copy()
    