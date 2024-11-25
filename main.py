import tkinter
import subprocess
import time
from model import Model
from customCheckpoint import ModelIntervalCheckpoint
from imageProcessor import ImageProcessor
import os, sys
import customtkinter as ctk
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gym
from tensorflow import keras
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
import pygame
from gym.utils.play import play
from pygame.surfarray import array3d
from snakeHuman.snake import Snake

IMG_SHAPE=(84,84)
WINDOW_LENGTH=4
class App(ctk.CTk):
    def __init__(window):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        window.geometry("600x500")
        window.after(0,lambda: window.wm_state('zoomed'))
        window.title("The gaming AI")
        window.start_snake=False
        window.highscore=0
        window.final_score=0.0
        window.file_name=''
        
        image_snake = ctk.CTkImage(light_image=Image.open("images/snake icon.png"), dark_image=Image.open("images/snake icon.png"), size=(40,40))
        image_cart=ctk.CTkImage(light_image=Image.open("images/cart_icon.png"), dark_image=Image.open("images/cart_icon.png"), size=(40,40))
        image_space=ctk.CTkImage(light_image=Image.open("images/spaceInvaders_icon.png"), dark_image=Image.open("images/spaceInvaders_icon.png"), size=(40,40))
        image_packman=ctk.CTkImage(light_image=Image.open("images/packman_icon.png"), dark_image=Image.open("images/packman_icon.png"), size=(40,40))
        image_raceCar = ctk.CTkImage(light_image=Image.open("images/racingcar_icon.png"), dark_image=Image.open("images/racingcar_icon.png"), size=(40,40))
        image_quit = ctk.CTkImage(light_image=Image.open("images/quit.png"), dark_image=Image.open("images/quit.png"), size=(40,40))
        image_snake_tab=ctk.CTkImage(light_image=Image.open("images/snake_tab.png"), dark_image=Image.open("images/snake_tab.png"), size=(250,250))

        # configure grid layout (4x4)
        window.grid_columnconfigure(1, weight=1)
        window.grid_columnconfigure((2, 3), weight=0)
        window.grid_rowconfigure((0, 1, 2,3), weight=1)

        #sidebar
        window.sidebar_frame=ctk.CTkFrame(window,width=140, corner_radius=0, fg_color="#708090")
        window.sidebar_frame.grid(row=0,column=0,rowspan=4, sticky="nsew")
        window.sidebar_frame.grid_rowconfigure(12, weight=1)

        window.sidebar_label_games=ctk.CTkLabel(window.sidebar_frame,text="Games", font=ctk.CTkFont(size=20,weight="bold"))
        window.sidebar_label_games.grid(row=0,column=0, padx=20, pady=(20,10))

        window.sidebar_button_snake=ctk.CTkButton(window.sidebar_frame,text="Sanke",image=image_snake,compound="right", command=window.sidebar_button_snake_event)
        window.sidebar_button_snake.grid(row=1, column=0, padx=20, pady=10)
        
        window.sidebar_button_snakeAI=ctk.CTkButton(window.sidebar_frame,text="Sanke AI",image=image_snake,compound="right", command=window.sidebar_button_snakeAI_event)
        window.sidebar_button_snakeAI.grid(row=2, column=0, padx=20, pady=10)
        
        window.sidebar_button_snake=ctk.CTkButton(window.sidebar_frame,text="Cart Pole",image=image_cart,compound="right", command=window.sidebar_button_cartPole_event)
        window.sidebar_button_snake.grid(row=3, column=0, padx=20, pady=10)
        
        window.sidebar_button_snakeAI=ctk.CTkButton(window.sidebar_frame,text="Cart Pole AI",image=image_cart,compound="right", command=window.sidebar_button_cartPoleAI_event)
        window.sidebar_button_snakeAI.grid(row=4, column=0, padx=20, pady=10)
        
        window.sidebar_button_snake=ctk.CTkButton(window.sidebar_frame,text="Space Invaders",image=image_space,compound="right", command=window.sidebar_button_spaceInvaders_event)
        window.sidebar_button_snake.grid(row=5, column=0, padx=20, pady=10)
        
        window.sidebar_button_snakeAI=ctk.CTkButton(window.sidebar_frame,text="Space Invaders AI",image=image_space,compound="right", command=window.sidebar_button_spaceInvadersAI_event)
        window.sidebar_button_snakeAI.grid(row=6, column=0, padx=20, pady=10)
        
        window.sidebar_button_snake=ctk.CTkButton(window.sidebar_frame,text="Packman",image=image_packman,compound="right", command=window.sidebar_button_packman_event)
        window.sidebar_button_snake.grid(row=7, column=0, padx=20, pady=10)
        
        window.sidebar_button_snakeAI=ctk.CTkButton(window.sidebar_frame,text="Packman AI",image=image_packman,compound="right", command=window.sidebar_button_packmanAI_event)
        window.sidebar_button_snakeAI.grid(row=8, column=0, padx=20, pady=10)
        
        window.sidebar_button_snake=ctk.CTkButton(window.sidebar_frame,text="Car Racing",image=image_raceCar,compound="right", command=window.sidebar_button_carRacing_event)
        window.sidebar_button_snake.grid(row=9, column=0, padx=20, pady=10)
        
        window.sidebar_button_snakeAI=ctk.CTkButton(window.sidebar_frame,text="Car Racing AI",image=image_raceCar,compound="right", command=window.sidebar_button_carRacingAI_event)
        window.sidebar_button_snakeAI.grid(row=10, column=0, padx=20, pady=10)
        
        

        window.sidebar_button_exit=ctk.CTkButton(window.sidebar_frame, text="Quit", image=image_quit, compound="right", command=window.sidebar_button_qiut_event)
        window.sidebar_button_exit.grid(row=11, column=0, padx=20, pady=10)


        window.sidebar_label_appereance=ctk.CTkLabel(window.sidebar_frame, text="Appearance Mode:", anchor="w")
        window.sidebar_label_appereance.grid(row=13, column=0, padx=20, pady=(10,0))
        window.sidebar_optionemenu_appmode=ctk.CTkOptionMenu(window.sidebar_frame, values=["Dark","Light"], command=window.change_appmode_event)
        window.sidebar_optionemenu_appmode.grid(row=14, column=0, padx=20, pady=(10,20))

        #Game preview
        window.game_frame=ctk.CTkFrame(window, width=100, fg_color="#708090")
        window.game_frame.grid(row=0, column=1, padx=(200, 200), pady=(20, 0),  sticky="nsew")
        window.game_frame.columnconfigure((0,1,2,3,4), weight=1)

        window.game_label_tittle=ctk.CTkLabel(window.game_frame,text="Home", font=ctk.CTkFont(size=20,weight="bold"))
        window.game_label_tittle.grid(row=0,column=2, padx=0, pady=(20,10))
        
        window.game_tabview=ctk.CTkTabview(window.game_frame, width=550)
        window.game_tabview.grid(row=1, column=2, padx=(20,0), pady=(20,0), sticky="nsew")
        window.game_tabview.add("Snake")
        window.game_tabview.add("Cart Pole")
        window.game_tabview.add("Space Invaders")
        window.game_tabview.add("Packman")
        window.game_tabview.add("Car Racing")
        window.game_tabview.tab("Snake").grid_columnconfigure(0,weight=1)
        window.game_tabview.tab("Cart Pole").grid_columnconfigure(0,weight=1)
        window.game_tabview.tab("Space Invaders").grid_columnconfigure(0,weight=1)
        window.game_tabview.tab("Packman").grid_columnconfigure(0,weight=1)
        window.game_tabview.tab("Car Racing").grid_columnconfigure(0,weight=1)
        
        window.game_tabview_snake_image=ctk.CTkLabel(window.game_tabview.tab("Snake"), text="Controls: W,A,S,D",)
        window.game_tabview_snake_image.grid(row=0,column=0, padx=0, pady=(20,20))
        window.generate_bar_chart(game="snakeScore",tab="Snake")
        weights_folder = "weights/snakeWeights"
        file_names = [os.path.splitext(f)[0] for f in os.listdir(weights_folder) if os.path.isfile(os.path.join(weights_folder, f))]
        window.snake_option_menu = ctk.CTkOptionMenu(window.game_tabview.tab("Snake"), values=file_names)
        window.snake_option_menu.grid(row=0, column=1, padx=0, pady=(10, 20))
        window.snake_option_menu.set(file_names[0])
        
        window.game_tabview_cart_pole_image=ctk.CTkLabel(window.game_tabview.tab("Cart Pole"), text="Controls: Left:A, Right:D", )
        window.game_tabview_cart_pole_image.grid(row=0,column=0, padx=0, pady=(20,20))
        window.generate_bar_chart(game="cartpoleScore",tab="Cart Pole")
        weights_folder = "weights/cartpoleWeights"
        file_names = [os.path.splitext(f)[0] for f in os.listdir(weights_folder) if os.path.isfile(os.path.join(weights_folder, f))]
        window.cartpole_option_menu = ctk.CTkOptionMenu(window.game_tabview.tab("Cart Pole"), values=file_names)
        window.cartpole_option_menu.grid(row=0, column=1, padx=0, pady=(10, 20))
        window.cartpole_option_menu.set(file_names[0])
        
        
        window.game_tabview_space_invaders_image=ctk.CTkLabel(window.game_tabview.tab("Space Invaders"), text="Controls: Left:A, Right:D, Shoot:SPACE", )
        window.game_tabview_space_invaders_image.grid(row=0, column=0, padx=0, pady=(20, 10))
        window.generate_bar_chart(game="spaceInvadersScore",tab="Space Invaders")
        weights_folder = "weights/spaceInvadersWeights"
        file_names = [os.path.splitext(f)[0] for f in os.listdir(weights_folder) if os.path.isfile(os.path.join(weights_folder, f))]
        window.space_invaders_option_menu = ctk.CTkOptionMenu(window.game_tabview.tab("Space Invaders"), values=file_names)
        window.space_invaders_option_menu.grid(row=0, column=1, padx=0, pady=(10, 20))
        window.space_invaders_option_menu.set(file_names[0])
        
        
        weights_folder = "weights/packmanWeights"
        window.game_tabview_packman_image=ctk.CTkLabel(window.game_tabview.tab("Packman"), text="Controls: W,A,S,D", )
        window.game_tabview_packman_image.grid(row=0,column=0, padx=0, pady=(20,20))
        window.generate_bar_chart(game="packmanScore",tab="Packman")
        file_names = [os.path.splitext(f)[0] for f in os.listdir(weights_folder) if os.path.isfile(os.path.join(weights_folder, f))]
        window.packman_option_menu = ctk.CTkOptionMenu(window.game_tabview.tab("Packman"), values=file_names)
        window.packman_option_menu.grid(row=0, column=1, padx=0, pady=(10, 20))
        window.packman_option_menu.set(file_names[0])
        
        window.game_tabview_car_racing_image=ctk.CTkLabel(window.game_tabview.tab("Car Racing"), text="Controls: W,A,S,D",)
        window.game_tabview_car_racing_image.grid(row=0,column=0, padx=0, pady=(20,20))
        window.generate_bar_chart(game="carRacingScore",tab="Car Racing")
        weights_folder = "weights/carRacingWeights"
        file_names = [os.path.splitext(f)[0] for f in os.listdir(weights_folder) if os.path.isfile(os.path.join(weights_folder, f))]
        window.car_racing_option_menu = ctk.CTkOptionMenu(window.game_tabview.tab("Car Racing"), values=file_names)
        window.car_racing_option_menu.grid(row=0, column=1, padx=0, pady=(10, 20))
        window.car_racing_option_menu.set(file_names[4])
        
        window.protocol("WM_DELETE_WINDOW", window.sidebar_button_qiut_event)
               
    
    def generate_bar_chart(window,game,tab):
        width=0.4
        # Read the number from the file
        with open("scores/"+game+".txt", "r") as file:
            scoreHuman = int(file.readline())
        with open("scores/"+game+"AI.txt", "r") as file:
            scoreAI = int(file.readline())
       # Plot the bar chart
        fig, ax = plt.subplots()
        ax.bar(["Player:"],[scoreHuman],width)
        ax.bar(["AI:"],[scoreAI],width)
        ax.set_ylabel("Value")
        ax.set_title("Highscore")
        
        # Convert the matplotlib figure to a Tkinter compatible image
        canvas = FigureCanvasTkAgg(fig, master=window.game_tabview.tab(tab))
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, padx=0, pady=(20,20))
    
    
    def on_done(window,results,file_name=''):
        highscore=0
        with open('scores/'+file_name+'ScoreAI.txt', 'r') as file:
            line = file.readline()
            highscore=int(line)
        for item in results:
            if item>highscore:
                highscore=item
                with open('scores/'+file_name+'ScoreAI.txt', 'w') as file:
                    file.write(str(int(item))) 
    
    def on_done_human(window, prev_obs, obs, action, rew, done, info):
        # Read highscore
        with open('scores/' + window.file_name + 'Score.txt', 'r') as file:
            line = file.readline()
            window.highscore = int(line)
        
        # Update final score
        window.final_score += rew
        
        # If the game is done, compare the score with the high score
        if done:
            print("Final Score:", window.final_score)
            if window.final_score > window.highscore:
                with open('scores/' + window.file_name + 'Score.txt', 'w') as file:
                    file.write(str(int(window.final_score)))
            # Reset the final score for the next game
            window.final_score = 0.0
       
        
    def sidebar_button_snake_event(window):
        BLACK = pygame.Color(0,0,0)
        WHITE=pygame.Color(255,255,255)
        RED=pygame.Color(255,0,0)
        GREEN=pygame.Color(0,255,0)
        snake_env=Snake(600,600)
        difficulty=10
        fps_controller=pygame.time.Clock()
        check_error=pygame.init()
        pygame.display.set_caption("Snake Game")

        while True:
            
            #Human Input
            for event in pygame.event.get():
                snake_env.action=snake_env.human_step(event)
            
            #Check direction
            snake_env.direction = snake_env.change_direction(snake_env.action, snake_env.direction)
            snake_env.snake_pos = snake_env.move(snake_env.direction, snake_env.snake_pos)
            
            #Check if we ate food
            snake_env.snake_body.insert(0,list(snake_env.snake_pos))
            if snake_env.eat():
                snake_env.score+=1
                snake_env.food_spawn=False
            else:
                snake_env.snake_body.pop()
            
            #Check if spawn new food
            if not snake_env.food_spawn:
                snake_env.food_pos=snake_env.spawn_food()
            snake_env.food_spawn=True
            
            #Drawing the snake
            snake_env.game_window.fill(BLACK)
            for pos in snake_env.snake_body:
                pygame.draw.rect(snake_env.game_window,GREEN, pygame.Rect(pos[0],pos[1],10,10))
            
            #Drawing the food
            pygame.draw.rect(snake_env.game_window,RED, pygame.Rect(snake_env.food_pos[0],snake_env.food_pos[1],10,10))
            
            
            # Check if endgame
            snake_env.game_over()            
            #window.on_done_human(None, None, None, snake_env.score, True, None, file_name='snake')
            
            #refresh game screen
            snake_env.display_score(WHITE,'arial',20)
            pygame.display.update()
            fps_controller.tick(difficulty)
            img=array3d(snake_env.game_window)
    
    def sidebar_button_snakeAI_event(window):
        env=gym.make("snake:snake-v0", render_mode="human",sleep=0.1)
        nb_actions = env.action_space.n
        if window.snake_option_menu.get()[:3] == "DUE":
            model=Model.build_dueling_model_atari(nb_actions)
        else:
            model=Model.build_model_atari(nb_actions)
        memory=SequentialMemory(limit=1000000,window_length=WINDOW_LENGTH)
        processor=ImageProcessor()
        policy=LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=0.2,value_min=0.1,value_test=0.05,nb_steps=1000000)
        dqn_agent=DQNAgent(model=model,
             nb_actions=nb_actions,
             policy=policy,
             memory=memory,
             processor=processor,
             nb_steps_warmup=50000,
             gamma=.99,
             target_model_update=10000,
             train_interval=4,
             delta_clip=1)
        if window.snake_option_menu.get()[:3] == "DOU":
            dqn_agent.enable_double_dqn=True
        dqn_agent.compile(Adam(learning_rate=0.00025),metrics=['mae'])
        model.load_weights("weights/snakeWeights/"+window.snake_option_menu.get()+".h5")
        results = dqn_agent.test(env, nb_episodes=1, visualize=False,)
        window.on_done(results.history["episode_reward"],file_name='snake')
        env.close()
        time.sleep(2)
        pygame.quit()
        
        
    def sidebar_button_cartPole_event(window):
        mapping = {(pygame.K_a,): 0, (pygame.K_d,): 1}
        window.highscore=0
        window.final_score=0.0
        window.file_name='cartpole'
        play(gym.make("CartPole-v1"), keys_to_action=mapping, seed=1,callback=window.on_done_human)
    
    def sidebar_button_cartPoleAI_event(window):
        env = gym.make("CartPole-v1",render_mode='human')
        obs_shape = env.observation_space.shape
        nb_actions = env.action_space.n
        model =Model.build_model_classic_control(obs_shape,nb_actions)
        dqn_agent = DQNAgent(
            model=model,
            memory=SequentialMemory(limit=50000, window_length=1),
            policy=BoltzmannQPolicy(),
            nb_actions=nb_actions,
            nb_steps_warmup=10,
            target_model_update=0.01
        )
        
        dqn_agent.compile(Adam(lr=0.001), metrics=["mae"])
        model.load_weights("weights/cartpoleWeights/"+window.cartpole_option_menu.get()+".h5")
        results = dqn_agent.test(env, nb_episodes=1, visualize=False,)
        window.on_done(results.history["episode_reward"],file_name='cartpole')
        env.close()
        
    def sidebar_button_spaceInvaders_event(window):
        mapping = {(pygame.K_0,):0, (pygame.K_SPACE,):1, (pygame.K_d,): 2, (pygame.K_a,): 3, (pygame.K_f,):4,(pygame.K_g,):5 }
        window.highscore=0
        window.final_score=0.0
        window.file_name='spaceInvaders'
        play(gym.make("SpaceInvaders-v0"), keys_to_action=mapping, zoom=3, callback=window.on_done_human)
    
    def sidebar_button_spaceInvadersAI_event(window):
        env = gym.make('ALE/SpaceInvaders-v5',full_action_space=False, render_mode='human')
        nb_actions = env.action_space.n
        if window.space_invaders_option_menu.get()[:3] == "DUE":
            model=Model.build_dueling_model_atari(nb_actions)
        else:
            model=Model.build_model_atari(nb_actions)
        memory=SequentialMemory(limit=1000000,window_length=WINDOW_LENGTH)
        processor=ImageProcessor()
        policy=LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=0.2,value_min=0.1,value_test=0.05,nb_steps=1000000)
        dqn_agent=DQNAgent(model=model,
             nb_actions=nb_actions,
             policy=policy,
             memory=memory,
             processor=processor,
             nb_steps_warmup=50000,
             gamma=.99,
             target_model_update=10000,
             train_interval=4,
             delta_clip=1)
        if window.space_invaders_option_menu.get()[:3] == "DOU":
            dqn_agent.enable_double_dqn=True
        dqn_agent.compile(Adam(learning_rate=0.00025),metrics=['mae'])
        model.load_weights("weights/spaceInvadersWeights/"+window.space_invaders_option_menu.get()+".h5")
        results = dqn_agent.test(env, nb_episodes=1, visualize=False,)
        window.on_done(results.history["episode_reward"],file_name='spaceInvaders')
        env.close()
        
        
    def sidebar_button_packman_event(window):
        mapping = {(pygame.K_0,):0, (pygame.K_w,):1, (pygame.K_d,): 2, (pygame.K_a,): 3, (pygame.K_s,):4}
        window.highscore=0
        window.final_score=0.0
        window.file_name='packman'
        play(gym.make("MsPacman-v0"), keys_to_action=mapping, zoom=3, callback=window.on_done_human)
    
    def sidebar_button_packmanAI_event(window):
        env = gym.make('ALE/MsPacman-v5',full_action_space=False, render_mode='human')
        nb_actions = env.action_space.n
        if window.packman_option_menu.get()[:3] == "DUE":
            model=Model.build_dueling_model_atari(nb_actions)
        else:
            model=Model.build_model_atari(nb_actions)
        memory=SequentialMemory(limit=1000000,window_length=WINDOW_LENGTH)
        processor=ImageProcessor()
        policy=LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=0.2,value_min=0.1,value_test=0.05,nb_steps=1000000)
        dqn_agent=DQNAgent(model=model,
             nb_actions=nb_actions,
             policy=policy,
             memory=memory,
             processor=processor,
             nb_steps_warmup=50000,
             gamma=.99,
             target_model_update=10000,
             train_interval=4,
             delta_clip=1)
        if window.packman_option_menu.get()[:3] == "DOU":
            dqn_agent.enable_double_dqn=True
        dqn_agent.compile(Adam(learning_rate=0.00025),metrics=['mae'])
        model.load_weights("weights/packmanWeights/"+window.packman_option_menu.get()+".h5")
        results = dqn_agent.test(env, nb_episodes=1, visualize=False,)
        window.on_done(results.history["episode_reward"],file_name='packman')
        env.close()
        
    def sidebar_button_carRacing_event(window):
        mapping = {(pygame.K_0,):0, (pygame.K_d,):1, (pygame.K_a,): 2, (pygame.K_w,): 3, (pygame.K_s,):4}
        window.highscore=0
        window.final_score=0.0
        window.file_name='carRacing'
        play(gym.make("CarRacing-v2", domain_randomize=True,continuous=False), keys_to_action=mapping, zoom=2, callback=window.on_done_human)
    
    def sidebar_button_carRacingAI_event(window):
        env = gym.make("CarRacing-v2", domain_randomize=True,continuous=False,render_mode='human')
        nb_actions = env.action_space.n
        if window.car_racing_option_menu.get()[:3] == "DUE":
            model=Model.build_dueling_model_atari(nb_actions)
        else:
            model=Model.build_model_atari(nb_actions)
        memory=SequentialMemory(limit=1000000,window_length=WINDOW_LENGTH)
        processor=ImageProcessor()
        policy=LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=0.2,value_min=0.1,value_test=0.05,nb_steps=1000000)
        dqn_agent=DQNAgent(model=model,
             nb_actions=nb_actions,
             policy=policy,
             memory=memory,
             processor=processor,
             nb_steps_warmup=50000,
             gamma=.99,
             target_model_update=10000,
             train_interval=4,
             delta_clip=1)
        if window.car_racing_option_menu.get()[:3] == "DOU":
            dqn_agent.enable_double_dqn=True
        dqn_agent.compile(Adam(learning_rate=0.00025),metrics=['mae'])
        model.load_weights("weights/carRacingWeights/"+window.car_racing_option_menu.get()+".h5")
        results = dqn_agent.test(env, nb_episodes=1, visualize=False,)
        window.on_done(results.history["episode_reward"],file_name='carRacing')
        env.close()

    
    def change_appmode_event(window,mode:str):
        ctk.set_appearance_mode(mode)
    
    def sidebar_button_qiut_event(window):
        plt.close('all')
        window.quit()
        window.destroy()


app=App()
app.mainloop()