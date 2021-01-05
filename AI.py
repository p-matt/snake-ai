import numpy as np
from random import randint
from snake import SnakeManager
from tqdm import tqdm

testGames, trainingGames, stepsPerGame = 0, 0, 0

best_score_train, total_score_train, total_step_train = 0, 0, 0
best_score, total_score, total_step = 0, 0, 0


def set_meta_parameters(test_games, training_games, steps_per_game):
    global testGames, trainingGames, stepsPerGame
    testGames, trainingGames, stepsPerGame = test_games, training_games, steps_per_game


def get_angle_with_apple(snake_position, apple_position):
    apple_vector = np.array(apple_position) - np.array(snake_position[-1])
    snake_vector = np.array(snake_position[-1]) - np.array(snake_position[-2])

    a = snake_vector / np.linalg.norm(snake_vector)
    # avoid division by zero
    if apple_position == snake_position[-1]:
        b = 0
    else:
        b = apple_vector / np.linalg.norm(apple_vector)

    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / np.pi


def get_directions_vectors(snake_position):
    current_direction_vector = (np.array(snake_position[-1]) - np.array(snake_position[-2]))
    left_direction_vector = np.array([current_direction_vector[-1], -current_direction_vector[-2]])
    right_direction_vector = np.array([-current_direction_vector[-1], current_direction_vector[-2]])
    return [current_direction_vector, left_direction_vector, right_direction_vector]


def get_observation(snake):
    directions_vectors = get_directions_vectors(snake.snakePositions)
    blocked_way = snake.direction_state(directions_vectors)
    angle = get_angle_with_apple(snake.snakePositions, snake.target)
    return np.array([blocked_way[0], blocked_way[1], blocked_way[2], angle])


def local_to_global(snakePos, local_dir):
    # local: -1> left; 0>straight; 1>right
    # global: 0>left; 1>right; 2>down; 3>up

    directions_vectors = get_directions_vectors(snakePos)
    current_direction_vector = directions_vectors[0]
    new_direction = current_direction_vector
    if local_dir == -1:
        new_direction = directions_vectors[1]
    elif local_dir == 1:
        new_direction = directions_vectors[2]

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


def get_random_directions(snakePos):
    local_dir = randint(-1, 1)
    global_dir = local_to_global(snakePos, local_dir)
    return local_dir, global_dir


def generate_dataset(display, clock):
    global best_score_train, total_score_train, total_step_train
    training_input = []

    for _ in tqdm(range(trainingGames)):
        snake = SnakeManager()
        prev_score = snake.score
        prev_observations = get_observation(snake)
        prev_food_distance = snake.get_food_distance()

        for current_step in range(stepsPerGame):
            local_dir, global_dir = get_random_directions(snake.snakePositions)
            snake.play(display, clock, global_dir)
            half_input = np.hstack((local_dir, prev_observations))

            # -1 when snake is dead
            if snake.is_dead():
                final_input = np.hstack((half_input, -1))
                training_input.append(final_input)
                break
            else:
                food_distance = snake.get_food_distance()
                reward = (620 - food_distance) / 620

                # +[0 to 1] when snake went closer from the food/ate the food depending the distance
                if snake.score > prev_score or food_distance <= prev_food_distance:
                    prev_score = snake.score
                    final_input = np.hstack((half_input, reward))
                    training_input.append(final_input)
                else:
                    # 0 when snake survived but wrong way from the food
                    final_input = np.hstack((half_input, 0))
                    training_input.append(final_input)

                prev_observations = get_observation(snake)
                prev_food_distance = food_distance
        # region debug
        if snake.score > best_score_train:
            best_score_train = snake.score
        total_step_train += current_step
        total_score_train += snake.score
        # endregion
    return np.array(training_input)


def think(display, clock, model):
    global best_score, total_score, total_step
    for _ in tqdm(range(testGames)):
        snake = SnakeManager()
        prev_observations = get_observation(snake)

        for current_step in range(stepsPerGame):

            predictions = []
            for local_dir in range(-1, 2):
                final_input = np.hstack((local_dir, prev_observations))
                predictions.append(model.predict(final_input.reshape(-1, 5, 1)))

            local_dir = np.argmax(np.array(predictions)) - 1
            global_dir = local_to_global(snake.snakePositions, local_dir)
            snake.play(display, clock, global_dir)

            if snake.is_dead():
                break
            else:
                prev_observations = get_observation(snake)

        if snake.score > best_score:
            best_score = snake.score

        # debug
        total_step += current_step
        total_score += snake.score
    debug()
    return


def debug():
    print(f"\nTRAINING")
    print(f"Score: best={best_score_train}, average={total_score_train / trainingGames}")
    print(f"Step: average={total_step_train / trainingGames}")

    print(f"\nRESULT")
    print(f"Score: best={best_score}, average={total_score / testGames}")
    print(f"Step: average={total_step / testGames}")
