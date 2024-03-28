from abc import ABC, abstractmethod

#all agents have to include basic functional.
class Agent(ABC):

    @abstractmethod
    def sample_action(self):
        ...
    
    @abstractmethod
    def train(self):
        ...
    
    @abstractmethod
    def interaction_step(self):
        ...
    
    @abstractmethod
    def save(self):
        ...