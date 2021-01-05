import random
import numpy as np
import pygame

newDataset, Render, frameRate = False, True, 50


class SnakeManager:

    def __init__(self):
        global newDataset
        self.snakeWidth = 20
        self.foodWidth = 20
        self.boundaries = (50, 670)
        self.isDead = False
        self.score = random.randint(3, 15) if newDataset else 3
        self.snakePositions, self.pos = self.spawn_snake()
        self.target = self.spawn_food()

    def spawn_snake(self):
        # spawn the snake depending his initial score
        head = 340 + (20 * (int(self.score / 2)))
        pos = []
        for i in range(self.score):
            pos.insert(0, [head - (i * 20), 340])
        return pos, [head, 340]

    def play(self, screen, clock, global_direction):
        isRunning = True
        while isRunning:
            # region handle pygame
            # keep the window open to watch the AI with this loop
            for evenement in pygame.event.get():
                if evenement.type == pygame.QUIT:
                    isRunning = False
            # endregion
            if Render:
                self.draw(screen)
                pygame.display.flip()
            self.new_turn(global_direction)
            clock.tick(frameRate)
            return

    def new_turn(self, global_dir):
        # global: 0>left; 1>right; 2>down; 3>up
        if global_dir == 0:
            self.pos[0] -= 20
        elif global_dir == 1:
            self.pos[0] += 20
        elif global_dir == 2:
            self.pos[1] += 20
        else:
            self.pos[1] -= 20

        self.update_snake()

        if self.pos == self.target:
            self.target = self.spawn_food()
            self.score += 1

    def update_snake(self):
        self.snakePositions.append(self.pos.copy())
        if len(self.snakePositions) > self.score: del self.snakePositions[0]

    def spawn_food(self):
        food = []
        while not len(food):
            food = [random.randrange(60, 660, 20), random.randrange(60, 660, 20)]
            if food in self.snakePositions:
                food = []
        return food

    def get_food_distance(self):
        return np.linalg.norm(np.array(self.target) - np.array(self.pos))

    # region pygame draw
    def draw(self, screen):
        screen.fill((0, 0, 0))

        pygame.draw.rect(screen, (255, 0, 0),
                         (self.target[0], self.target[1], self.foodWidth, self.foodWidth))

        self.draw_snake(screen)

        self.draw_border(screen)

        self.set_text(40, 'Snake Game', (250, 10, 100, 50), (255, 255, 255), screen)
        self.set_text(35, '{}'.format(str(self.score)), (500, 10, 100, 50), (255, 255, 255), screen)

    def draw_snake(self, screen):
        for pos in self.snakePositions[:-1]:
            pygame.draw.rect(screen, (0, 255, 0), (pos[0], pos[1], self.snakeWidth, self.snakeWidth))
        pygame.draw.rect(screen, (255, 255, 0), (self.pos[0], self.pos[1], self.snakeWidth, self.snakeWidth))

    @staticmethod
    def draw_border(screen):
        pygame.draw.rect(screen, (255, 255, 255), (60, 60, 630, 630), 3)

    # endregion

    # region collision
    def collision_with_self(self, head):
        return head in self.snakePositions[:-1]

    def collision_with_boundaries(self, pos):
        return pos[0] < self.boundaries[0] or pos[0] > self.boundaries[1] or pos[1] < self.boundaries[0] or pos[1] > \
               self.boundaries[1]

    def is_direction_closed(self, n_dir):
        next_step = self.pos + n_dir
        return self.collision_with_boundaries(next_step) or self.collision_with_self(next_step.tolist())

    def direction_state(self, directions_vectors):
        # 0>safe, 1>blocked
        current_direction_vector = directions_vectors[0]
        left_direction_vector = directions_vectors[1]
        right_direction_vector = directions_vectors[2]

        is_front_closed = 1 if self.is_direction_closed(current_direction_vector) else 0
        is_left_closed = 1 if self.is_direction_closed(left_direction_vector) else 0
        is_right_closed = 1 if self.is_direction_closed(right_direction_vector) else 0

        return [is_left_closed, is_front_closed, is_right_closed]

    def is_dead(self):
        return self.collision_with_boundaries(self.pos) or self.collision_with_self(self.pos)

    # endregion

    @staticmethod
    def set_meta_parameters(new_dataset, render, frame_rate):
        global newDataset, Render, frameRate
        newDataset = new_dataset
        Render = render
        frameRate = frame_rate

    @staticmethod
    def set_text(size, message, message_rectangle, color, screen):
        font = pygame.font.SysFont('Lato', size, False)
        message = font.render(message, True, color)
        screen.blit(message, message_rectangle)
