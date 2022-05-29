import time

cardinals_dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
nb_step = 0
size = 0
maze = []


class Node:

    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.f = 0
        self.g = 0
        self.h = 0
        self.child = []

    def set_value(self, h):
        self.f = self.g + h
        self.h = h


def is_solved(current_node, end_node):
    if current_node.position == end_node.position:  # le noeud courant est localisé en position d'arrivée
        path = []  # on récupère/retourne tous ses noeuds parents, ce qui correspond au chemin de sortie
        current = current_node
        while current:
            path.append(current.position)
            current = current.parent
        return path
    return False


def DFS_get_next_nodes(current_node, end_pos):
    global nb_step
    next_nodes = []  # liste des prochains nodes à retourner
    positions = get_valids_pos(current_node.position[0], current_node.position[1])  # positions pour le dfs

    parent = current_node
    local_g = 0

    while positions:
        nb_step += 1
        local_g += 1
        y, x = positions.pop(-1)

        if is_a_node(y, x, end_pos):
            new_node = Node((y, x), parent)
            new_node.g = local_g
            next_nodes.append(new_node)
            parent = current_node
            local_g = 0
        else:
            n = Node((y, x), parent)
            parent = n
            finding_next_node = get_valids_pos(y, x)
            if finding_next_node is not None:
                positions.append(finding_next_node[0])
            else:
                parent = current_node
                local_g = 0
        maze[y][x][0] = 1

    return next_nodes


def is_a_node(y, x, end_pos):
    nb_empty = 0
    for dir in cardinals_dir:
        new_pos = (y + dir[0], x + dir[1])
        if new_pos[0] > size - 1 or new_pos[0] < 0 or new_pos[1] > size - 1 or new_pos[1] < 0:  # on vérifie qu'on soit dans le labyrinthe
            continue
        if maze[new_pos[0]][new_pos[1]][0] == 0 or maze[new_pos[0]][new_pos[1]][0] == 1:
            nb_empty += 1
    return nb_empty > 2 or [y, x] == end_pos


def get_valids_pos(y, x):
    valids_pos = []
    for dir in cardinals_dir:
        new_pos = (y + dir[0], x + dir[1])
        if new_pos[0] > size - 1 or new_pos[0] < 0 or new_pos[1] > size - 1 or new_pos[
            1] < 0:  # on vérifie qu'on soit dans le labyrinthe
            continue
        elif maze[new_pos[0]][new_pos[1]][0] == 1:  # on vérifie qu'on est sur un chemin accessible (pas sur le snake)
            continue
        elif maze[new_pos[0]][new_pos[1]][1]:  # on vérifie que le noeud ne fait pas parti du chemin existant
            continue

        maze[y][x][1] = True  # cellule visité
        valids_pos.append(new_pos)

    if len(valids_pos):
        return valids_pos
    return None


def AStar(start_pos, end_pos):
    startNode = Node(start_pos)
    endNode = Node(end_pos)
    nodes = [startNode]  # contient tous les noeuds à explorer

    while nodes:
        current_node = sorted(nodes, key=lambda x: x.f)[0]
        del nodes[nodes.index(current_node)]
        solved = is_solved(current_node, endNode)
        if solved:
            return solved

        next_nodes = DFS_get_next_nodes(current_node, end_pos)
        for next_node in next_nodes:  # assignation des valeurs f g h
            h = abs(endNode.position[0] - next_node.position[0]) + abs(endNode.position[1] - next_node.position[1])
            next_node.set_value(h)
            nodes.append(next_node)
    return False


def pathfinding(board, start_pos, end_pos):
    global size, maze
    s = time.time()
    size = len(board)
    maze = board

    path = AStar(start_pos, end_pos)

    if path:
        return path[:-1]
    else:
        return None
