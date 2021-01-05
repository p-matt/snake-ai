import AI as AI
import tflearn
from snake import *
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import seaborn as sns
# network
neurons = 100
batchSize = 8
epochs = 3
learningRate = .0075

# AI
testGames = 100
trainingGames = 20000
stepsPerGame = 2500


def get_model():
    network = input_data(shape=[None, 5, 1], name='input')
    network = fully_connected(network, neurons, activation='relu')
    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=learningRate, batch_size=batchSize,
                         loss='mean_square', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


def train_model(data, model):
    X = data[:, 0:-1].reshape(-1, 5, 1)
    y = data[:, -1].reshape(-1, 1)
    model.fit(X, y, n_epoch=epochs, shuffle=True, run_id="model_and_weights", show_metric=False, snapshot_epoch=False, snapshot_step=10**5)
    model.save("save/model_n20k.tfl")
    return model


def start():
    new_dataset = input("\nGenerate new DataSet and train the AI ? (y/n)").lower() == "y"
    model = get_model()
    AI.set_meta_parameters(testGames, trainingGames, stepsPerGame)
    # region pygame
    pygame.init()
    screen = pygame.display.set_mode((750, 750))
    pygame.display.set_caption('SnAIke')
    clock = pygame.time.Clock()

    # endregion

    if new_dataset:
        SnakeManager.set_meta_parameters(new_dataset=new_dataset, render=False, frame_rate=10**6)
        training_data = AI.generate_dataset(screen, clock)
        trained_model = train_model(training_data, model)
        SnakeManager.set_meta_parameters(new_dataset=False, render=False, frame_rate=10**6)
        AI.think(screen, clock, trained_model)
    else:
        SnakeManager.set_meta_parameters(new_dataset=new_dataset, render=True, frame_rate=50)
        model.load("save/model_n10k.tfl")
        AI.think(screen, clock, model)


start()