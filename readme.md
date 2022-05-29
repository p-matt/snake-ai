# AI plays Snake Game
## Reinforcement learning

### Implementation : 
1. Generate a Dataset from games
2. Train the neural net
3. Test the model  
  

## 1. Generate a Dataset  
First thing is to get a 2D DataSet, where there are 12 features:
- 7 of these features implies distances to the closest wall in 7 directions (from 0 to ~40)
- Angle between snake’s direction and direction to the food (from -1 to 1)
- One hot encoded suggested local direction (-1 — left, 0 — forward, 1 — right)
- Manhattan distance between snake and food

Label of these features is defined arbitrarily depending if the snake is alive or not and if he went closer from the food or not.  
- Snake **died** from his last move: **y = -1**  
- Snake is still **alive** but **further** from the food: **y = 0**  
- Snake is still **alive** and **closer** from the food: **y = [0,1] : (D - food_distance) / D**
where D is equal to the hypotenuse of the playing area. This way we can tell how good was the situation relatively to the distance between snake and food.    
 
The generation of the Dataset is effective through n trainings games where the snake move randomly and his first size is random between 2 integers.    
I also implemented a generation of dataset that involves A* algorithm and doesn't make random move anymore.   
The final DataSet looks like :

## 2. Train the Neural Network 
I'm using keras to implement the NN.  
NN architecture:
- 3 hidden layer (64 > 32 > 16)  
- Dropout and batch normalization is used after the non-linearity (ReLU here)  

From all the training data injected in the neural network, it will be able to evaluate the quality of a given situation and this is the output of the NN.  
Following this process the snake will not move randomly anymore, he simply takes the way that has the maximum output value from the NN which correspond to the best situation.  


## 3. Testing
For the same Dataset structure, minimizing the loss function (MSE) is very dependents to the quantity and the quality of data. With only few examples the system will be very dependent of these specific situations.
  
<img width="80%" src="https://user-images.githubusercontent.com/74459226/103469589-0b362480-4d67-11eb-9c87-afaf4c0b61cc.png"/>

nb. of example |10      | 10'000     
:---: | :---: |  :---: |
preview |<img src="https://user-images.githubusercontent.com/74459226/103469881-fa87ad80-4d6a-11eb-818b-f92c7c4ace19.gif"/> | <img src="https://user-images.githubusercontent.com/74459226/103469879-f6f42680-4d6a-11eb-9d97-14efd62f6adc.gif"/>

## Sources
https://github.com/m-tosch/Snake-AI/blob/master/README.md  
https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3   
https://theailearner.com/2018/04/19/snake-game-with-deep-learning/''  