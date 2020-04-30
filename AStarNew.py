from __future__ import print_function

class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):

    startNode = Node(None, start)
    startNode.g = 0
    startNode.h = 0
    startNode.f = 0

    endNode = Node(None, end)
    endNode.g = 0
    endNode.h = 0
    endNode.f = 0

    openList = []
    closedList = []

    openList.append(startNode)

    while len(openList) > 0:

        currentNode = openList[0]
        currentIndex = 0
        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIndex = index

        openList.pop(currentIndex)
        closedList.append(currentNode)

        if currentNode == endNode:
            path = []
            current = currentNode
            current2 = currentNode
            count = 0
            while current is not None:
                path.append(current.position)
                current = current.parent
            while current2 is not None:
                current2 = current2.parent
                if count < len(path)-1:
                    nodePosition = (current2.position[0],current2.position[1])
                    maze[nodePosition[0]][nodePosition[1]] = 3
                    count += 1
                nodePosition = (currentNode.position[0],currentNode.position[1])
                maze[nodePosition[0]][nodePosition[1]] = 3
            return path[::-1]

        #Generate neighbours for the current node
        neighbours  = []
        for newPosition in [(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]:

            nodePosition = (currentNode.position[0] + newPosition[0], currentNode.position[1]+newPosition[1])

            if nodePosition[0] >(len(maze)-1) or nodePosition[0] < 0 or nodePosition[1] > (len(maze[len(maze)-1])-1) or nodePosition[1] < 0:
                continue

            if maze[nodePosition[0]][nodePosition[1]] != 0:
                continue

            newNode = Node(currentNode, nodePosition)

            neighbours.append(newNode)

        for neighbour in neighbours:
            for closedNeighbour in closedList:
                if neighbour == closedNeighbour:
                    continue
            neighbour.g = currentNode.g + 1
            neighbour.h = ((neighbour.position[0]-endNode.position[0])**2)+((neighbour.position[1]-endNode.position[1])**2)
            neighbour.f = neighbour.g + neighbour.h

            for openNode in openList:
                if neighbour == openNode and neighbour.g > openNode.g:
                    continue

            openList.append(neighbour)

def main():

    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],

            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],

            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]]


    start = (5, 1)

    end = (1, 1)

    path = astar(maze, start, end)




    print(*maze, sep='\n')
    print("")
    print(path)




if __name__ == '__main__':

    main()
