from gridworld import GridWorld
import numpy as np

class WindyGridWorld(GridWorld):
    def __init__(self, shape, goal_positions, wind_vector, boundary_penalty=-1, time_penalty=0, goal_reward=1, agent_pos=None):
        super(WindyGridWorld, self).__init__(shape, goal_positions, boundary_penalty, time_penalty, goal_reward, agent_pos)
        assert len(wind_vector) == shape[1], 'Wind vector size is not equal to map row size'

    @classmethod
    def frommatrix(cls, matrix, wind_vector):
        gpos = [(index, row.index(1)) for index, row in enumerate(matrix) if 1 in row]
        return cls(np.array(matrix).shape, gpos, wind_vector)


if __name__ == '__main__':
    gw = WindyGridWorld((6,5), [(3,4)], [1, 1, 1, 1, 1])
    print(gw.map)

    mapm = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]

    gw2 = WindyGridWorld.frommatrix(mapm, [1, 1, 1, 1])
    print(gw2.map)