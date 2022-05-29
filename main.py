import AI as AI
from snake import *
from utils import *

# AI
testGames = 100
trainingGames = 100
steps_max_per_game = 6_000
n_generation = 1
new_data = False

epochs = 250


def train(screen, clock):
    model = load_model()

    for i in range(n_generation):
        X, y = AI.generate_dataset(screen, clock, model)
        model.fit(X, y, validation_split=.1, epochs=epochs, callbacks=[checkpoint_callback("model", i + 1), es_callback])
        save(X, y, i + 1)


def start():
    AI.set_parameters(testGames, trainingGames, steps_max_per_game)
    if new_data:
        print("- generating new data")
        screen, clock = init_pygame()

        SnakeManager.set_parameters(new_dataset=True, render=False, frame_rate=10 ** 6, debug=False)
        # X, y = AI.generate_dataset(screen, clock, None)
        # save(X, y)
        train(screen, clock)
    else:
        screen, clock = init_pygame()
        SnakeManager.set_parameters(new_dataset=False, render=True, frame_rate=50, debug=False)
        model = load_model("1")
        AI.think(screen, clock, model)


start()
