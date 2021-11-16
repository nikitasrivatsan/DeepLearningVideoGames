#!/usr/bin/env python
#Modified from http://www.pygame.org/project-Very+simple+Pong+game-816-.html

import numpy
import pygame
import os
from pygame.locals import *
from sys import exit
import random
import pygame.surfarray as surfarray
import matplotlib.pyplot as plt

position = 5, 325
os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
pygame.init()
screen = pygame.display.set_mode((640,480),0,32)
#screen = pygame.display.set_mode((640,480),pygame.NOFRAME)
#Creating 2 bars, a ball and background.
back = pygame.Surface((640,480))
background = back.convert()
background.fill((0,0,0))
bar = pygame.Surface((10,50))
bar1 = bar.convert()
bar1.fill((0,255,255))
bar2 = bar.convert()
bar2.fill((255,255,255))
circ_sur = pygame.Surface((15,15))
circ = pygame.draw.circle(circ_sur,(255,255,255),(15//2,15//2),15//2)
circle = circ_sur.convert()
circle.set_colorkey((0,0,0))
font = pygame.font.SysFont("calibri",40)


ai_speed = 15.

HIT_REWARD = 0
LOSE_REWARD = -1
SCORE_REWARD = 1

class GameState:
    def __init__(self):
        self.bar1_x, self.bar2_x = 10. , 620.
        self.bar1_y, self.bar2_y = 215. , 215.
        self.circle_x, self.circle_y = 307.5, 232.5
        self.bar1_move, self.bar2_move = 0. , 0.
        self.bar1_score, self.bar2_score = 0,0
        self.speed_x, self.speed_y = 7., 7.

    def frame_step(self,input_vect):
        pygame.event.pump()
        reward = 0

        if sum(input_vect) != 1:
            raise ValueError('Multiple input actions!')

        if input_vect[1] == 1:#Key up
            self.bar1_move = -ai_speed
        elif input_vect[2] == 1:#Key down
            self.bar1_move = ai_speed
        else: # don't move
            self.bar1_move = 0
                
        self.score1 = font.render(str(self.bar1_score), True,(255,255,255))
        self.score2 = font.render(str(self.bar2_score), True,(255,255,255))

        screen.blit(background,(0,0))
        frame = pygame.draw.rect(screen,(255,255,255),Rect((5,5),(630,470)),2)
        middle_line = pygame.draw.aaline(screen,(255,255,255),(330,5),(330,475))
        screen.blit(bar1,(self.bar1_x,self.bar1_y))
        screen.blit(bar2,(self.bar2_x,self.bar2_y))
        screen.blit(circle,(self.circle_x,self.circle_y))
        screen.blit(self.score1,(250.,210.))
        screen.blit(self.score2,(380.,210.))

        self.bar1_y += self.bar1_move
        
        #AI of the computer.
        if self.circle_x >= 305.:
            if not self.bar2_y == self.circle_y + 7.5:
                if self.bar2_y < self.circle_y + 7.5:
                    self.bar2_y += ai_speed
                if  self.bar2_y > self.circle_y - 42.5:
                    self.bar2_y -= ai_speed
            else:
                self.bar2_y == self.circle_y + 7.5
        
        # bounds of movement
        if self.bar1_y >= 420.: self.bar1_y = 420.
        elif self.bar1_y <= 10. : self.bar1_y = 10.
        if self.bar2_y >= 420.: self.bar2_y = 420.
        elif self.bar2_y <= 10.: self.bar2_y = 10.

        #since i don't know anything about collision, ball hitting bars goes like this.
        if self.circle_x <= self.bar1_x + 10.:
            if self.circle_y >= self.bar1_y - 7.5 and self.circle_y <= self.bar1_y + 42.5:
                self.circle_x = 20.
                self.speed_x = -self.speed_x
                reward = HIT_REWARD

        if self.circle_x >= self.bar2_x - 15.:
            if self.circle_y >= self.bar2_y - 7.5 and self.circle_y <= self.bar2_y + 42.5:
                self.circle_x = 605.
                self.speed_x = -self.speed_x

        # scoring
        if self.circle_x < 5.:
            self.bar2_score += 1
            reward = LOSE_REWARD
            self.circle_x, self.circle_y = 320., 232.5
            self.bar1_y,self.bar_2_y = 215., 215.
        elif self.circle_x > 620.:
            self.bar1_score += 1
            reward = SCORE_REWARD
            self.circle_x, self.circle_y = 307.5, 232.5
            self.bar1_y, self.bar2_y = 215., 215.

        # collisions on sides
        if self.circle_y <= 10.:
            self.speed_y = -self.speed_y
            self.circle_y = 10.
        elif self.circle_y >= 457.5:
            self.speed_y = -self.speed_y
            self.circle_y = 457.5

        self.circle_x += self.speed_x
        self.circle_y += self.speed_y

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()

        terminal = False
        if max(self.bar1_score, self.bar2_score) >= 20:
            self.bar1_score = 0
            self.bar2_score = 0
            terminal = True

        return image_data, reward, terminal
