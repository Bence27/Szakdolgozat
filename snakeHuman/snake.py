import pygame, sys, time, random
from pygame.surfarray import array3d


BLACK = pygame.Color(0,0,0)
WHITE=pygame.Color(255,255,255)
RED=pygame.Color(255,0,0)
GREEN=pygame.Color(0,255,0)

class Snake():
    def __init__(self,frame_size_x,frame_size_y):
        self.frame_size_x=frame_size_x
        self.frame_size_y=frame_size_y
        self.game_window=pygame.display.set_mode((frame_size_x,frame_size_y))
        
        #Reset The GAme
        self.reset()
    
    
    def reset(self):
        self.game_window.fill(BLACK)
        self.snake_pos=[100,50]
        self.snake_body=[[100,50],[100-10,50],[100-20,50]]
        self.food_pos=self.spawn_food()
        self.food_spawn=True
        
        self.direction='RIGHT'
        self.action=self.direction
        
        self.score=0
        self.steps=0
        print('GAME RESET')
    
        
    def spawn_food(self):
        return [random.randrange(1,(self.frame_size_x//10))*10,random.randrange(1,(self.frame_size_y//10))*10]
        
    
    def eat(self):
        return self.snake_pos[0]==self.food_pos[0]and self.snake_pos[1]==self.food_pos[1]
    
    def change_direction(self,action,direction):
        if action=='UP' and direction !='DOWN':
            direction='UP'
        if action=='DOWN' and direction !='UP':
            direction='DOWN'
        if action=='LEFT' and direction !='RIGHT':
            direction='LEFT'
        if action=='RIGHT' and direction !='LEFT':
            direction='RIGHT'
            
        return direction
    
    def move(self,direction,snake_pos):
        if direction=='UP':
            snake_pos[1]-=10
        if direction=='DOWN':
            snake_pos[1]+=10
        if direction=='LEFT':
            snake_pos[0]-=10
        if direction=='RIGHT':
            snake_pos[0]+=10
        
        return snake_pos
    
    def human_step(self,event):
        action=None
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_UP:
                action='UP'
            if event.key==pygame.K_DOWN:
                action='DOWN'
            if event.key==pygame.K_LEFT:
                action='LEFT'
            if event.key==pygame.K_RIGHT:
                action='RIGHT'
            if event.key==pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        return action
    
    def display_score(self,color,font,size):
        
        score_font=pygame.font.SysFont(font,size)
        score_surface=score_font.render("Score: "+str(self.score),True,color)
        
        score_rect=score_surface.get_rect()
        score_rect.midtop=(self.frame_size_x/10,15)
        self.game_window.blit(score_surface,score_rect)
        
    def game_over(self):
        #Touch box
        if self.snake_pos[0]<0 or self.snake_pos[0]>self.frame_size_x-10:
            self.end_game()

        if self.snake_pos[1]<0 or self.snake_pos[1]>self.frame_size_y-10:
            self.end_game()
       
        #Touch own body
        for block in self.snake_body[1:]:
            if self.snake_pos[0]==block[0] and self.snake_pos[1]==block[1]:
                self.end_game()
    
    def end_game(self):
        message=pygame.font.SysFont('arial',45)
        message_surface=message.render('GAME OVER',True,WHITE)
        message_rect=message_surface.get_rect()
        message_rect.midtop=(self.frame_size_x/2,self.frame_size_y/4)
        self.game_window.fill(BLACK)
        self.game_window.blit(message_surface,message_rect)
        self.display_score(RED, 'arial',20)
        pygame.display.flip()
        time.sleep(2)
        pygame.quit()
        

            