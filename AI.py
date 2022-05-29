import numpy as np
from random import randint
from snake import SnakeManager
from tqdm import tqdm
from astar import pathfinding

testGames, trainingGames, steps_max_per_game = 0, 0, 0

best_score_train, total_score_train, total_step_train = 0, 0, 0
best_score, total_score, total_step = 0, 0, 0

ohe_local_dir = {-1: [0, 0, 1], 0: [0, 1, 0], 1: [1, 0, 0]}

distance_const = 876.8


def set_parameters(test_games, training_games, steps_per_game):
    global testGames, trainingGames, steps_max_per_game
    testGames, trainingGames, steps_max_per_game = test_games, training_games, steps_per_game


def get_observation(snake):
    directions_state = snake.direction_state(snake.get_directions_vectors())
    angle = snake.get_angle_with_apple()
    path = pathfinding(snake.get_matrix_field(), snake.current_pos(), snake.target_pos())
    if path and SnakeManager.Render:
        snake.draw_path(path)
    path_count = len(path) if path else -1

    return np.array([path_count, angle] + directions_state)


def local_to_global(snake, local_dir):
    # local: -1> left; 0>straight; 1>right
    # global: 0>left; 1>right; 2>down; 3>up

    directions_vectors = snake.get_directions_vectors()
    if local_dir == 0:
        new_direction = snake.current_dir_vect
    elif local_dir == -1:
        new_direction = directions_vectors[0]
    elif local_dir == 1:
        new_direction = directions_vectors[1]
    else:
        print("error here")

    new_direction = new_direction / 20
    if [-1, 0] == new_direction.tolist():
        global_dir = 0
    elif [1, 0] == new_direction.tolist():
        global_dir = 1
    elif [0, 1] == new_direction.tolist():
        global_dir = 2
    else:
        global_dir = 3
    return global_dir


def get_random_directions(snake):
    local_dir = randint(-1, 1)
    global_dir = local_to_global(snake, local_dir)
    return local_dir, global_dir


def play(snake, model, obs):
    if model:
        predictions = []
        for local_dir_candidates in range(-1, 2):
            ld_ohe = ohe_local_dir[local_dir_candidates]
            X = np.hstack((ld_ohe, obs))
            predictions.append(model.predict(X.reshape(1, -1)))

        local_dir = np.argmax(np.array(predictions)) - 1
        global_dir = local_to_global(snake, local_dir)
    else:
        local_dir, global_dir = get_random_directions(snake)
    snake.play(global_dir)

    return local_dir


def generate_dataset(display, clock, model=None):
    global best_score_train, total_score_train, total_step_train
    X = []
    y = []

    for _ in tqdm(range(trainingGames)):
        snake = SnakeManager(display, clock)
        index_dead = 0
        for current_step in range(steps_max_per_game):
            # get obs
            observations = get_observation(snake)
            snake_score = snake.score
            food_distance = snake.get_food_distance()

            # play
            local_dir = play(snake, model, observations)

            # -1 when snake is dead
            if snake.is_dead():
                X.append(np.hstack((ohe_local_dir[local_dir], observations)))
                y.append(-1)
                break
            else:
                X.append(np.hstack((ohe_local_dir[local_dir], observations)))

                # +[0 to 1] when snake went closer from the food/ate the food depending the distance
                if snake.score > snake_score:
                    y.append(1)
                else:
                    new_food_distance = snake.get_food_distance()
                    if new_food_distance < food_distance:
                        reward = (distance_const - snake.get_food_distance()) / distance_const
                        y.append(reward)
                    else:
                        y.append(0)

        if snake.score > best_score_train:
            best_score_train = snake.score
        total_step_train += current_step
        total_score_train += snake.score
    debug()
    return np.array(X), np.array(y)


def think(display, clock, model=None):
    global best_score, total_score, total_step
    if model is not None:
        for _ in tqdm(range(testGames)):
            snake = SnakeManager(display, clock)
            for current_step in range(steps_max_per_game):
                observations = get_observation(snake)
                predictions = []
                for local_dir_candidates in range(-1, 2):
                    ld_ohe = ohe_local_dir[local_dir_candidates]
                    X = np.hstack((ld_ohe, observations)).reshape(1, -1)
                    predictions.append(model.predict(X))
                local_dir = np.argmax(np.array(predictions)) - 1
                global_dir = local_to_global(snake, local_dir)
                snake.play(global_dir)
                if snake.is_dead():
                    # print(predictions)
                    # input()
                    break

            if snake.score > best_score:
                best_score = snake.score

            # debug
            total_step += current_step
            total_score += snake.score
    else:
        snake = SnakeManager()
        while not snake.is_dead():
            path = pathfinding(snake.get_matrix_field(), snake.current_pos(), snake.target_pos())
            if path and SnakeManager.Render:
                snake.draw_path(path, display)
            for dir in directions:
                snake.play(display, clock, dir)

    # debug()


def debug():
    global best_score_train, total_score_train, total_step_train, best_score, total_score, total_step
    print(f"\nTRAINING")
    print(f"Score: best={best_score_train}, average={total_score_train / trainingGames}")
    print(f"Step: average={total_step_train / trainingGames}")

    print(f"\nRESULT")
    print(f"Score: best={best_score}, average={total_score / testGames}")
    print(f"Step: average={total_step / testGames}")

    best_score_train, total_score_train, total_step_train = 0, 0, 0
    best_score, total_score, total_step = 0, 0, 0
