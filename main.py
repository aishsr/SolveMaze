# Assignment 1 - Q6
# Author: Aish Srinivas
# Date: 30 May 2021
# Spring 2021

import numpy as np
from collections import deque
from operator import attrgetter
import matplotlib.pyplot as plt
import copy
import time

max_rows = 20
max_columns = 20


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class PointWithHeuristic:
    def __init__(self, parent=None, cell=None):
        self.parent = parent
        self.cell = cell

        self.g = 0
        self.h = 0
        self.f = 0


def checkIfValid(row, column):
    if (row >= 0) and (row < max_rows) and (column >= 0) and (column < max_columns):
        return True
    return False


def BFS(maze, source: Point, destination: Point):
    rowNum = [1, 0, -1, 0]
    colNum = [0, 1, 0, -1]

    if maze[source.x][source.y] == 1 or maze[destination.x][destination.y] == 1:
        return -1

    visited = [[False for i in range(max_columns)] for j in range(max_rows)]
    visited[source.x][source.y] = True

    q = deque()
    q.append([source, 0, source])
    path = []

    while q:
        current = q.popleft()

        point_to_look_at = current
        path.append(point_to_look_at)

        if point_to_look_at[0].x == destination.x and point_to_look_at[0].y == destination.y:

            new_path = []
            current_node = current[0]
            new_path.append((current_node.x, current_node.y))

            while current_node != source:
                for i in path:
                    if current_node.x == i[0].x and current_node.y == i[0].y:
                        current_node = i[2]
                        new_path.append((current_node.x, current_node.y))
            new_path.pop()
            # for i in path:
            #     print("(" + str(i[0].x) + ", " + str(i[0].y) + ")")
            return list(reversed(new_path)), current[1], path

        for i in range(4):
            row = point_to_look_at[0].x + rowNum[i]
            col = point_to_look_at[0].y + colNum[i]

            if checkIfValid(row, col) and not visited[row][col] and maze[row][col] == 0:
                visited[row][col] = True
                q.append([Point(row, col), current[1] + 1, Point(point_to_look_at[0].x, point_to_look_at[0].y)])
    return -1


def DFS(maze, source: Point, dest: Point):
    rowNum = [1, 0, -1, 0]
    colNum = [0, 1, 0, -1]

    if maze[source.x][source.y] == 1 or maze[dest.x][dest.y] == 1:
        return -1

    visited = [[False for i in range(max_columns)] for j in range(max_rows)]
    visited[source.x][source.y] = True

    stack = deque()
    stack.append([source, 0, source])
    path = []

    while stack:
        current = stack.pop()
        point_to_look_at = current
        path.append(point_to_look_at)

        if point_to_look_at[0].x == dest.x and point_to_look_at[0].y == dest.y:

            new_path = []
            current_node = current[0]
            new_path.append((current_node.x, current_node.y))
            while current_node != source:
                for i in path:
                    if current_node.x == i[0].x and current_node.y == i[0].y:
                        current_node = i[2]
                        new_path.append((current_node.x, current_node.y))
            new_path.pop()
            return list(reversed(new_path)), current[1], path

        for i in range(4):
            row = point_to_look_at[0].x + rowNum[i]
            col = point_to_look_at[0].y + colNum[i]

            if checkIfValid(row, col) and not visited[row][col] and maze[row][col] == 0:
                visited[row][col] = True
                stack.append([Point(row, col), current[1] + 1, Point(point_to_look_at[0].x, point_to_look_at[0].y)])
    return -1


def A_star(maze, source: Point, dest: Point, start_time):
    start_h = PointWithHeuristic(None, (source.x, source.y))
    dest_h = PointWithHeuristic(None, (dest.x, dest.y))

    opened = []
    closed = []
    rowNum = [0, 0, -1, 1]
    colNum = [-1, 1, 0, 0]
    opened.append(start_h)

    # Loop until you find the end
    while opened:
        opened.sort(key=attrgetter('f'), reverse=False)
        current_node = opened.pop(0)
        print(str(current_node.cell[0]) + ", " + str(current_node.cell[1]))
        closed.append(current_node)

        if current_node.cell[0] == dest_h.cell[0] and current_node.cell[1] == dest_h.cell[1]:
            path = []
            total_g = 0
            c = current_node
            while c is not None:
                path.append(c.cell)
                c = c.parent
                total_g += 1
            return list(reversed(path)), total_g, opened, time.perf_counter() - start_time

        for i in range(4):

            row = current_node.cell[0] + rowNum[i]
            col = current_node.cell[1] + colNum[i]
            node_cell = (row, col)

            if not checkIfValid(row, col) or (checkIfValid(row, col) and maze[row][col] == 1):
                continue

            child = PointWithHeuristic(current_node, node_cell)

            flag = 0
            for c in closed:
                if child.cell == c.cell:
                    flag += 1
                    continue
            for o in opened:
                if child.cell == o.cell and o.g < child.g:
                    flag += 1
                    continue

            if flag == 0:
                child.g = current_node.g + 1
                child.h = ((abs(child.cell[0] - dest_h.cell[0])) ** 2) + (
                        (abs(child.cell[1] - dest_h.cell[1])) ** 2)
                child.f = child.g + child.h

                opened.append(child)


def dijkstra(maze, source: Point, dest: Point, start_time):
    start_h = PointWithHeuristic(None, (source.x, source.y))
    dest_h = PointWithHeuristic(None, (dest.x, dest.y))

    opened = []
    closed = []
    rowNum = [0, 0, -1, 1]
    colNum = [-1, 1, 0, 0]
    opened.append(start_h)

    # Loop until you find the end
    while opened:
        opened.sort(key=attrgetter('f'), reverse=False)
        current_node = opened.pop(0)
        closed.append(current_node)

        if current_node.cell[0] == dest_h.cell[0] and current_node.cell[1] == dest_h.cell[1]:
            path = []
            total_g = 0
            c = current_node
            while c is not None:
                path.append(c.cell)
                c = c.parent
                total_g += 1
            return list(reversed(path)), total_g, opened, time.perf_counter() - start_time

        for i in range(4):

            row = current_node.cell[0] + rowNum[i]
            col = current_node.cell[1] + colNum[i]
            node_cell = (row, col)

            if not checkIfValid(row, col) or (checkIfValid(row, col) and maze[row][col] == 1):
                continue

            child = PointWithHeuristic(current_node, node_cell)

            flag = 0
            for c in closed:
                if child.cell == c.cell:
                    flag += 1
                    continue
            for o in opened:
                if child.cell == o.cell and o.g < child.g:
                    flag += 1
                    continue

            if flag == 0:
                child.g = current_node.g + 1
                child.h = 0
                child.f = child.g + child.h

                opened.append(child)


def buildGraph(source, destination, maze, optimal_path, plot_number, traversed, cost):
    alternate_maze = maze
    if 7 <= plot_number <= 9:
        for i in traversed:
            alternate_maze[i.cell[0]][i.cell[1]] = 0.8
    else:
        for i in traversed:
            alternate_maze[i[0].x][i[0].y] = 0.6

    for i in optimal_path:
        alternate_maze[i[0]][i[1]] = 0.4

    alternate_maze[source.x][source.y] = 0.3
    alternate_maze[destination.x][destination.y] = 0.2

    return alternate_maze


def main():
    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # creating graphical view of maze
    n_maze = np.array(maze)
    n_maze = n_maze.astype(np.float64)
    for i in range(20):
        for j in range(20):
            n_maze[i][j] = not n_maze[i][j]

    plt.imshow(n_maze, cmap="gray")
    plt.savefig('maze.png')

    # # our points
    # start = Point(0, 0)
    # # e1 = Point(19, 23)
    # e2 = Point(0, 8)

    start_alternate = Point(0, 0)
    end = Point(24, 24)

    # # Implementing the search methods
    # print("By using BFS:")
    #
    # # ################################################################ Path from S to E1
    # dist_e1_BFS = BFS(maze, start, e1)
    # print("Path from S to E1 = " + str(dist_e1_BFS[0]))
    # print("Cost of path = " + str(dist_e1_BFS[1]))
    # print("Number of nodes explored = " + str(len(dist_e1_BFS[2])))
    # p_1_maze = copy.deepcopy(n_maze)
    # m1 = buildGraph(start, e1, p_1_maze, dist_e1_BFS[0], 1, dist_e1_BFS[2], dist_e1_BFS[1])
    #
    # plt.figure(1)
    # plt.title("Cost of path = " + str(dist_e1_BFS[1]))
    # plt.imshow(m1, cmap="gray")
    # plt.savefig('BFS_S_E1.png')
    #
    # print(" ")
    #
    # # ################################################################ Path from S to E2
    # dist_e2_BFS = BFS(maze, start, e2)
    # print("Path from S to E2 = " + str(dist_e2_BFS[0]))
    # print("Cost of path = " + str(dist_e2_BFS[1]))
    # print("Number of nodes explored = " + str(len(dist_e2_BFS[2])))
    # p_2_maze = copy.deepcopy(n_maze)
    # m2 = buildGraph(start, e2, p_2_maze, dist_e2_BFS[0], 2, dist_e2_BFS[2], dist_e2_BFS[1])
    #
    # plt.figure(2)
    # plt.title("Cost of path = " + str(dist_e2_BFS[1]))
    # plt.imshow(m2, cmap="gray")
    # plt.savefig('BFS_S_E2.png')
    #
    # print(" ")
    #
    # # ############################################################### Path from (0, 0) to (24, 24)
    # dist_full_BFS = BFS(maze, start_alternate, end)
    # print("Path from (0, 0) to (24, 24) = " + str(dist_full_BFS[0]))
    # print("Cost of path = " + str(dist_full_BFS[1]))
    # print("Number of nodes explored = " + str(len(dist_full_BFS[2])))
    # p_3_maze = copy.deepcopy(n_maze)
    # m3 = buildGraph(start_alternate, end, p_3_maze, dist_full_BFS[0], 3, dist_full_BFS[2], dist_full_BFS[1])
    #
    # plot3 = plt.figure(3)
    # plt.title("Cost of path = " + str(dist_full_BFS[1]))
    # plt.imshow(m3, cmap="gray")
    # plt.savefig('BFS_fullmaze.png')
    #
    # print(" ")
    #
    # print("By using DFS:")
    #
    # # ################################################################ Path from S to E2
    # dist_e1_DFS = DFS(maze, start, e1)
    # print("Path from S to E1 = " + str(dist_e1_DFS[0]))
    # print("Cost of path = " + str(dist_e1_DFS[1]))
    # print("Number of nodes explored = " + str(len(dist_e1_DFS[2])))
    # p_4_maze = copy.deepcopy(n_maze)
    # m4 = buildGraph(start, e1, p_4_maze, dist_e1_DFS[0], 4, dist_e1_DFS[2], dist_e1_DFS[1])
    #
    # plot4 = plt.figure(4)
    # plt.title("Cost of path = " + str(dist_e1_DFS[1]))
    # plt.imshow(m4, cmap="gray")
    # plt.savefig('DFS_S_E1.png')
    #
    # print(" ")
    #
    # # ################################################################ Path from S to E2
    # dist_e2_DFS = DFS(maze, start, e2)
    # print("Path from S to E2 = " + str(dist_e2_DFS[0]))
    # print("Cost of path = " + str(dist_e2_DFS[1]))
    # print("Number of nodes explored = " + str(len(dist_e2_DFS[2])))
    # p_5_maze = copy.deepcopy(n_maze)
    # m5 = buildGraph(start, e2, p_5_maze, dist_e2_DFS[0], 5, dist_e2_DFS[2], dist_e2_DFS[1])
    #
    # plot5 = plt.figure(5)
    # plt.title("Cost of path = " + str(dist_e2_DFS[1]))
    # plt.imshow(m5, cmap="gray")
    # plt.savefig('DFS_S_E2.png')
    #
    # print(" ")
    #
    # # ############################################################### Path from (0, 0) to (24, 24)
    # dist_full_DFS = DFS(maze, start_alternate, end)
    # print("Path from (0, 0) to (24, 24) = " + str(dist_full_DFS[0]))
    # print("Cost of path = " + str(dist_full_DFS[1]))
    # print("Number of nodes explored = " + str(len(dist_full_DFS[2])))
    # p_6_maze = copy.deepcopy(n_maze)
    # m6 = buildGraph(start_alternate, end, p_6_maze, dist_full_DFS[0], 6, dist_full_DFS[2], dist_full_DFS[1])
    #
    # plot6 = plt.figure(6)
    # plt.title("Cost of path = " + str(dist_full_DFS[1]))
    # plt.imshow(m6, cmap="gray")
    # plt.savefig('DFS_fullmaze.png')
    #
    # print(" ")
    #
    # print("By using A*:")
    #
    # ################################################################ Path from S to E1
    #
    start = Point(1, 2)
    e1 = Point(12, 3)

    start_time = time.perf_counter()

    # dist_e1_A = A_star(maze, start, e1, start_time)
    # print("Path from S to D = " + str(dist_e1_A[0]))
    # print("Cost of path = " + str(dist_e1_A[1]))
    # print("Number of nodes explored = " + str(len(dist_e1_A[2])))
    # print("Time taken = " + str(dist_e1_A[3]))
    #
    # p_7_maze = copy.deepcopy(n_maze)
    # m7 = buildGraph(start, e1, p_7_maze, dist_e1_A[0], 7, dist_e1_A[2], dist_e1_A[1])
    #
    # plot7 = plt.figure(7)
    # plt.title("Cost of path = " + str(dist_e1_A[1]))
    # plt.imshow(m7, cmap="gray")
    # plt.savefig('A_fullmaze.png')
    #
    # print(" ")
    #
    # # ################################################################ Path from S to E1
    # dist_d = dijkstra(maze, start, e1, start_time)
    # print("Path from S to D = " + str(dist_d[0]))
    # print("Cost of path = " + str(dist_d[1]))
    # print("Number of nodes explored = " + str(len(dist_d[2])))
    # print("Time taken = " + str(dist_d[3]))
    #
    # p_8_maze = copy.deepcopy(n_maze)
    # m8 = buildGraph(start, e1, p_8_maze, dist_d[0], 8, dist_d[2], dist_d[1])
    #
    # plot7 = plt.figure(8)
    # plt.title("Cost of path = " + str(dist_d[1]))
    # plt.imshow(m8, cmap="gray")
    # plt.savefig('dijkstra_fullmaze.png')
    #
    # print(" ")
    #

    # ################################################################ Path from S to E2
    # dist_e2_A = A_star(maze, start, e2, start_time)
    # print("Path from S to E2 = " + str(dist_e2_A[0]))
    # print("Cost of path = " + str(dist_e2_A[1]))
    # print("Number of nodes explored = " + str(len(dist_e2_A[2])))
    # p_8_maze = copy.deepcopy(n_maze)
    # m8 = buildGraph(start, e2, p_8_maze, dist_e2_A[0], 8, dist_e2_A[2], dist_e2_A[1])
    #
    # plot8 = plt.figure(8)
    # plt.title("Cost of path = " + str(dist_e2_A[1]))
    # plt.imshow(m8, cmap="gray")
    # plt.savefig('A_S_E2.png')
    #
    # print(" ")

    # ############################################################### Path from (0, 0) to (24, 24)

    start = Point(0, 0)
    end = Point(0, 8)

    dist_full_A = A_star(maze, start, end, start_time)
    print("Path from (0, 0) to (24, 24) = " + str(dist_full_A[0]))
    print("Cost of path = " + str(dist_full_A[1]))
    print("Number of nodes explored = " + str(len(dist_full_A[2])))
    p_9_maze = copy.deepcopy(n_maze)
    m9 = buildGraph(start, end, p_9_maze, dist_full_A[0], 7, dist_full_A[2], dist_full_A[1])

    plot9 = plt.figure(7)
    plt.title("Cost of path = " + str(dist_full_A[1]))
    plt.imshow(m9, cmap="gray")
    plt.savefig('A_fullmaze.png')

    print(" ")


main()
