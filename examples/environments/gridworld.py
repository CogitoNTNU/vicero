import numpy as np

# Based on the GridWorld environment defined in S&B18

class GridWorld:
    NORTH, SOUTH, EAST, WEST = range(4)

    def __init__(self, shape, goal_positions, boundary_penalty=-1, time_penalty=0, goal_reward=1, agent_pos=None):
        
        # shape and goal_position are on a (row, column) format

        self.map = np.zeros(shape)
        for goal_position in goal_positions:
            assert goal_position[0] in range(shape[0]) and \
                   goal_position[1] in range(shape[1]), 'Goal position is outside grid!'
            self.map[goal_position[0]][goal_position[1]] = 1
    
        self.agent_pos = (np.random.randint(0, self.map.shape[0]), np.random.randint(0, self.map.shape[1])) if agent_pos is None else agent_pos
        self.action_space = [self.NORTH, self.SOUTH, self.EAST, self.WEST]
        self.state_space = range(shape[0] * shape[1])
        self.boundary_penalty = boundary_penalty
        self.time_penalty = time_penalty
        self.goal_reward = goal_reward


    @classmethod
    def frommatrix(cls, matrix):
        gpos = [(index, row.index(1)) for index, row in enumerate(matrix) if 1 in row]
        return cls(np.array(matrix).shape, gpos)

    def step(self, action):
        new_pos = self.agent_pos

        if action == self.NORTH: new_pos = (max(0, min(new_pos[0] - 1, self.map.shape[0] - 1)), new_pos[1])
        if action == self.SOUTH: new_pos = (max(0, min(new_pos[0] + 1, self.map.shape[0] - 1)), new_pos[1])
        if action == self.EAST: new_pos = (new_pos[0], max(0, min(new_pos[1] + 1, self.map.shape[1] - 1)))
        if action == self.WEST: new_pos = (new_pos[0], max(0, min(new_pos[1] - 1, self.map.shape[1] - 1)))
        
        reward = self.time_penalty if new_pos != self.agent_pos else self.boundary_penalty

        self.agent_pos = new_pos

        #return state, reward, done, info 
        return 0, reward, False, {}


if __name__ == '__main__':
    gw = GridWorld((3, 4), [(2, 3)], agent_pos=(0, 0))
    print(gw.step(GridWorld.EAST))
    print(gw.step(GridWorld.EAST))
    print(gw.step(GridWorld.EAST))
    print(gw.step(GridWorld.EAST))
    print(gw.step(GridWorld.SOUTH))
    print(gw.step(GridWorld.SOUTH))
    print(gw.step(GridWorld.SOUTH))
    
    """
    gw = GridWorld((6,5), [(3,4)])
    print(gw.map)

    mapm = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]
    gw2 = GridWorld.frommatrix(mapm)
    print(gw2.map)
    """