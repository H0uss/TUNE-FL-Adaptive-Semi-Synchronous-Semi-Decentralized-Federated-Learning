from math import isqrt
import copy
import networkx as nx
import random
import numpy as np
import torch

from trainer import train
from utils import check_small_elements



class NetworkGraph:
    def __init__(self, num_nodes:int , graph_format:str):
        self.num_nodes = num_nodes
        self.graph_format = graph_format
        self.graph = None
        self.num_message_passing = None
        
    def create_graph(self):
        self.graph = nx.Graph()
        
        if self.graph_format == "ring":
            self.graph = nx.cycle_graph(self.num_nodes)
        elif self.graph_format == "fully_connected":
            self.graph = nx.complete_graph(self.num_nodes)
        elif self.graph_format == "mesh":
            self.graph = self.create_mesh_graph(self.num_nodes)
        elif self.graph_format == "partially_connected":
            self.create_partially_connected_topology()
        elif self.graph_format == "random":
            self.graph = self.create_connected_graph(self.num_nodes)
        else:
            raise ValueError("Invalid graph format")
    
    def create_partially_connected_topology(self):
        # A fully connected graph firstly is created, then we remove edges randomly.
        self.graph = nx.complete_graph(self.num_nodes)
        edges = list(self.graph.edges())
        num_edges_to_remove = random.randint(0, len(edges))
        edges_to_remove = random.sample(edges, num_edges_to_remove)
        self.graph.remove_edges_from(edges_to_remove)

    def create_mesh_graph(self, num_nodes):
        G = nx.Graph()
        
        # Compute the grid dimensions
        side = isqrt(num_nodes)  # Largest integer sqrt of N
        
        for i in range(num_nodes):
            G.add_node(i)
        
        # Connect nodes in a 2D grid pattern
        for i in range(num_nodes):
            if (i + 1) % side != 0 and i + 1 < num_nodes:  # Connect right (avoid wrapping)
                G.add_edge(i, i + 1)
            if i + side < num_nodes:  # Connect downward
                G.add_edge(i, i + side)
        return G
    
    def create_connected_graph(self, nodes):
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(range(nodes))

        # Generate a random number of edges
        num_edges = random.randint(nodes - 1, nodes * (nodes - 1) // 2)

        # Add random edges
        while G.number_of_edges() < num_edges:
            node1 = random.randint(0, nodes - 1)
            node2 = random.randint(0, nodes - 1)
            if node1 != node2 and not G.has_edge(node1, node2):
                G.add_edge(node1, node2)

        # Ensure no isolated nodes
        while nx.number_of_isolates(G) > 0:
            isolates = list(nx.isolates(G))
            node = random.choice(isolates)
            connected_node = random.choice(list(G.nodes()))
            G.add_edge(node, connected_node)

        return G
    
    def calculate_sum_sequence_adj_matrix(self,n):
        try: 
            A = nx.adjacency_matrix(self.graph).todense()
            m = A.shape[0]  #
            I = np.eye(m)   # Identity matrix 

            # Calculate A^(n+1)
            An = np.linalg.matrix_power(A, n + 1)

            # Calculate (A^(n+1) - I)
            An_minus_I = An - I

            # Calculate inverse of (A - I)
            A_minus_I_inv = np.linalg.inv(A - I)

            # Inverse(A-I) * (A^(n+1) - I)
            result = np.dot(A_minus_I_inv, An_minus_I)

            return result
        
        except:
            A = nx.adjacency_matrix(self.graph).todense() # calculate the sum using a for loop in case A^n is not invertible 
            sum_A = 0
            for i in range(1,n+1):
                sum_A += np.linalg.matrix_power(A, i)
            return
        
    def calculate_num_message_passing(self):
        low = 1
        high = self.num_nodes 

        while low < high:
            mid = (low + high) // 2
            sum_sequence = self.calculate_sum_sequence_adj_matrix(mid)
            result = check_small_elements(sum_sequence)
        
            if result == False:
                low = mid + 1 # move to the right
            else:
                high = mid # move to the left

        self.num_message_passing = low



class Client():
    def __init__(self, id, edge_server ,trainloader, valloader, model, threshold, cfg):
        self.id = id # client id
        self.edge_server = edge_server # to which edge server this client belongs

        self.trainloader = trainloader
        self.valloader = valloader
        self.data_size = len(trainloader.dataset)
        self.model = copy.deepcopy(model)
        self.base_model = copy.deepcopy(model)
        self.agg_weight = 0

        self.threshold = threshold
        self.cfg = cfg
        self.next_round_time = float('inf')

        self.is_malicious = False
        self.selection_counter = 0
        self.is_selected = 0

        self.finish_counter = [1]

    def fit(self, criterion, fl_round, round_time, device):
        self.base_model = copy.deepcopy(self.model)
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.train_config['lr'],
            momentum=self.cfg.train_config['momentum'],
            weight_decay=self.cfg.train_config['weight_decay']
                                    )   
        epsilon = self.cfg.train_config['epsilon']
        end_time, delayed_weight, num_iterations  = train(self.model, criterion , self.valloader ,optim, self.cfg.sleep_time 
                          ,self.threshold, fl_round ,round_time, epsilon ,device)
        
        
        
        return end_time, delayed_weight, num_iterations
        
    def set_parameters(self, model):
        self.model.load_state_dict(copy.deepcopy(model))

    def get_parameters(self):
        return self.model.state_dict()
    
    def set_round_time(self, estimation):
        self.next_round_time = estimation


        

def spawn_clients(trainloaders, valloaders ,thresholds ,model,cfg):
    id = 0
    node = 0
    clients = []
    for trainloader_, valloader_, threshold_ in zip(trainloaders,valloaders, thresholds):
        cc = [Client(id+i, node ,trainloader_[i],valloader_[i], model ,threshold_[i],cfg) for i in range(len(trainloader_))]
        id+=len(trainloader_)
        node+=1
        clients.append(cc)

    return clients



    