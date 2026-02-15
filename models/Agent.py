import torch

class EntityState(object):

    def __init__(self, position=None, velocity=None):
        self.velocity = velocity
        self.position = position


class AgentState(EntityState):

    def __init__(self):
        super(AgentState, self).__init__()

        self.comm_range = None
        self.curr_obs = None


class Action(object):

    def __init__(self):
        # Physical action
        self.physical = None
        # Communication action
        self.communication = None


class Entity():

    def __init__(self, x, y, width, height, n_mol_types=None):
        super(Entity, self).__init__()

        self.id = None

        self.collide = True

        self.max_velocity = None
        self.acceleration = None

        self.n_mol_types = n_mol_types

        self.dimension = [width, height]

        self.state = EntityState(position=[x, y], velocity=[0, 0])

    def move(self, move):

        self.state.position[0] += move[0]
        self.state.position[1] += move[1]


class Agent(Entity):
    def __init__(self,
                 max_mol_out,
                 n_mol_types,
                 comm_range=None,
                 device='cpu'):
        super().__init__(0, 0, 0, 0, n_mol_types=n_mol_types)

        self.state = AgentState()
        self.state.comm_range = comm_range
        self.n_mol_types = n_mol_types
        self.max_mol_out = max_mol_out

        self.device = device

        self.state.curr_obs = torch.zeros(self.n_mol_types)