import random
import numpy as np
import pygame


class SnakeManager:
    newDataset, Render, frameRate = False, True, 1
    Debug = False

    def __init__(self, screen, clock):
        self.snakeWidth = 20
        self.foodWidth = 20
        self.boundaries = (50, 670)
        self.isDead = False
        self.score = random.randint(3, 20) if SnakeManager.newDataset else 3
        self.snakePositions, self.head = self.spawn_snake()
        self.target = self.spawn_food()
        self.current_dir_vect = np.array(self.head) - np.array(self.snakePositions[-2])
        self.screen = screen
        self.clock = clock
        self.render()

    def spawn_snake(self):
        # spawn the snake depending his initial score
        head = 340 + (20 * (int(self.score / 2)))
        pos = []
        for i in range(self.score):
            pos.insert(0, [head - (i * 20), 340])
        return pos, [head, 340]

    def play(self, global_direction):
        isRunning = True
        while isRunning:
            # region pygame
            # keep the window open to watch the AI with this loop
            for evenement in pygame.event.get():
                if evenement.type == pygame.QUIT:
                    isRunning = False
            # endregion

            if self.new_turn(global_direction):
                self.render()
                self.clock.tick(SnakeManager.frameRate)

            return

    items = {"empty": 0, "snake": 1, "target": 2, "visited": False}

    def render(self, force=False):
        if SnakeManager.Render or force:
            self.draw()
            if force:
                m = self.get_matrix_field()
                for y in m:
                    r = ""
                    for x in y:
                        r += str(x[0])
                    print(r)

    def get_matrix_field(self):
        # 0 => empty
        # 1 => snake
        # 2 => target
        m = [[[SnakeManager.items["empty"], SnakeManager.items["empty"]] for w in range(31)] for h in range(31)]
        for (x, y) in self.snakePositions:
            _x, _y = int(x / 20) - 3, int(y / 20) - 3
            m[_y][_x][0] = SnakeManager.items["snake"]
            m[_y][_x][1] = SnakeManager.items["visited"]
        m[int(self.target[1] / 20) - 3][int(self.target[0] / 20) - 3][0] = SnakeManager.items["target"]
        return m

    def current_pos(self):
        return int(self.head[1] / 20) - 3, int(self.head[0] / 20) - 3

    def target_pos(self):
        return int(self.target[1] / 20) - 3, int(self.target[0] / 20) - 3

    def new_turn(self, global_dir):
        # global: 0>left; 1>right; 2>down; 3>up
        if global_dir == 0:
            self.head[0] -= 20
        elif global_dir == 1:
            self.head[0] += 20
        elif global_dir == 2:
            self.head[1] += 20
        else:
            self.head[1] -= 20

        if self.is_dead():
            self.isDead = True
            return False
        else:
            self.update_snake()
            self.set_current_direction_vector()

            if self.head == self.target:
                self.target = self.spawn_food()
                self.score += 1
            return True

    def update_snake(self):
        self.snakePositions.append(self.head.copy())
        if len(self.snakePositions) > self.score:
            del self.snakePositions[0]

    def spawn_food(self):
        food = []
        while not food:
            food = [random.randrange(60, 660, 20), random.randrange(60, 660, 20)]
            if food in self.snakePositions:
                food = []
        return food

    def get_food_distance(self):
        return np.sqrt((self.target[0] - self.head[0]) ** 2 + (self.target[1] - self.head[1]) ** 2)

    def set_current_direction_vector(self):
        self.current_dir_vect = np.array(self.head) - np.array(self.snakePositions[-2])

    def get_angle_with_apple(self):
        apple_vector = np.array(self.target) - np.array(self.head)
        a = self.current_dir_vect / np.linalg.norm(self.current_dir_vect)
        # avoid division by zero
        if self.target == self.head:
            b = 0
        else:
            b = apple_vector / np.linalg.norm(apple_vector)

        return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / np.pi

    def collision_with_self(self, head):
        return head in self.snakePositions[:-1]

    def is_out_of_zone(self, pos):
        return pos[0] < self.boundaries[0] or pos[0] > self.boundaries[1] or pos[1] < self.boundaries[0] or pos[1] > \
               self.boundaries[1]

    def distance_from_collision(self, n_dir):
        for i in range(1, 50):
            next_step = self.head + (n_dir * i)
            if self.is_out_of_zone(next_step) or self.collision_with_self(next_step.tolist()):
                return i - 1
            if SnakeManager.Render and SnakeManager.Debug:
                self.draw_position(next_step)

    def get_directions_vectors(self):
        left_direction_vector = np.array([self.current_dir_vect[-1], -self.current_dir_vect[-2]])
        right_direction_vector = np.array([-self.current_dir_vect[-1], self.current_dir_vect[-2]])
        return [left_direction_vector, right_direction_vector]

    def direction_state(self, directions_vectors):
        # 0>safe, 1>blocked
        left_direction_vector = directions_vectors[0]
        right_direction_vector = directions_vectors[1]
        front_left_direction_vector = self.current_dir_vect + left_direction_vector
        front_right_direction_vector = self.current_dir_vect + right_direction_vector
        back_left_direction_vector = -self.current_dir_vect + left_direction_vector
        back_right_direction_vector = -self.current_dir_vect + right_direction_vector

        left_distance = self.distance_from_collision(left_direction_vector)
        front_left_distance = self.distance_from_collision(front_left_direction_vector)
        front_distance = self.distance_from_collision(self.current_dir_vect)
        front_right_distance = self.distance_from_collision(front_right_direction_vector)
        right_distance = self.distance_from_collision(right_direction_vector)
        back_left_distance = self.distance_from_collision(back_left_direction_vector)
        back_right_distance = self.distance_from_collision(back_right_direction_vector)
        return [left_distance, front_left_distance, front_distance, front_right_distance, right_distance,
                back_left_distance, back_right_distance]

    def is_dead(self):
        return self.is_out_of_zone(self.head) or self.collision_with_self(self.head)

    # region pygame draw
    def draw(self):
        self.screen.fill((0, 0, 0))

        pygame.draw.rect(self.screen, (255, 0, 0), (self.target[0], self.target[1], self.foodWidth, self.foodWidth))

        self.draw_snake()

        self.draw_border()

        self.set_text(40, 'Snake Game', (250, 10, 100, 50), (255, 255, 255))
        self.set_text(35, '{}'.format(str(self.score)), (500, 10, 100, 50), (255, 255, 255))
        pygame.display.flip()

    def draw_snake(self):
        for pos in self.snakePositions[:-1]:
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[0], pos[1], self.snakeWidth, self.snakeWidth))
        pygame.draw.rect(self.screen, (255, 255, 0), (self.head[0], self.head[1], self.snakeWidth, self.snakeWidth))

    def draw_position(self, position):
        pygame.draw.rect(self.screen, (0, 255, 255), (position[0], position[1], self.snakeWidth, self.snakeWidth))
        pygame.time.delay(1)
        pygame.display.flip()

    def draw_path(self, path):
        if SnakeManager.Debug:
            for pos in path[::-1][:-1]:
                x, y = ((pos[1] + 3) * 20), int((pos[0] + 3) * 20)
                pygame.draw.rect(self.screen, (0, 255, 255), (x, y, self.snakeWidth, self.snakeWidth))
                pygame.time.delay(1)
                pygame.display.flip()

    def draw_border(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (60, 60, 630, 630), 3)

    # endregion

    @staticmethod
    def set_parameters(new_dataset, render, frame_rate, debug):
        SnakeManager.newDataset = new_dataset
        SnakeManager.Render = render
        SnakeManager.frameRate = frame_rate
        SnakeManager.Debug = debug

    def set_text(self, size, message, message_rectangle, color):
        font = pygame.font.SysFont('Lato', size, False)
        message = font.render(message, True, color)
        self.screen.blit(message, message_rectangle)
