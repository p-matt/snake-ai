# AI plays Snake Game - Neural Network
## Supervised learning

### Implementation : 
1. Generate a DataSet from games
2. Train the Neural Network with the DataSet
3. Test the model
&nbsp; 
## 1. Generate a DataSet  
First thing is to get a 2D DataSet, where there are 5 **Features** (X):
- Is left closed (1 — yes, 0 — no)
- Is front closed (1 — yes, 0 — no)
- Is right closed (1 — yes, 0 — no)
- Normalized angle between snake’s direction and direction to the food (from -1 to 1)
- Suggested local direction (-1 — left, 0 — forward, 1 — right)

The **Label** (Y) of these features is defined arbitrarily depending if the snake is alive or not and if he went closer from the food or not.  
- If the snake **died** from his last move: **y = -1**  
- If the snake is still **alive** but **further** from the food: **y = 0**  
- If the snake is still **alive** and **closer** from the food: **y = [0,1] : (D - food_distance) / D**
where D is equal to the size of one side of the playing area. This way we can tell if the situation was favorable or not and also how good it was.    
 
The generation of the DataSet is effective through **n** trainings games where the snake move randomly and his first size is random between 2 integers.    

The final DataSet looks like :  

X1      | X2     | X3    | X4    | X5    | Y
| :---: |  :---: | :---: | :---: | :---: | :---: 
True    | True   | False | 0.1   | 1     | `0.5`
False   | True   | False | -0.5  | 0     | `-1`
True    | False  | True  | 0.7   | -1    | `0`

## 2. Train the Neural Network 
All the training part is fit using **TFlearn** which is an high API of Tensorflow.  
From all the training data injected in the neural network, it will be able to **estimate/evaluate the quality of any move** and this is the **output** of the NN.  
Following this process the snake will not move randomly anymore, he simply takes the way that has the maximum output value from the NN.  
&nbsp;
## 3. Testing
For the same DataSet structure, the **final veracity** of the AI is **very dependent of number of exemple** (number of trainings games). With only few examples the system will be very dependent of these specific situations. With more examples there will be better result with longer learning phase while the system is not overfed.  
The **final veracity** is also very dependent from other factor such as the **learning rate**, **the way the training data is generated** (from random moves? from random size ? ...) , the number of neurons etc.
There is how the varacity of the AI varies:  
  
  
<img width="80%" src="https://user-images.githubusercontent.com/74459226/103469589-0b362480-4d67-11eb-9c87-afaf4c0b61cc.png"/>

nb. of example |10      | 10'000     
:---: | :---: |  :---: |
preview |<img src="https://user-images.githubusercontent.com/74459226/103469881-fa87ad80-4d6a-11eb-818b-f92c7c4ace19.gif"/> | <img src="https://user-images.githubusercontent.com/74459226/103469879-f6f42680-4d6a-11eb-9d97-14efd62f6adc.gif"/>

## Sources
https://github.com/m-tosch/Snake-AI/blob/master/README.md  
https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3   
https://theailearner.com/2018/04/19/snake-game-with-deep-learning/''  