
import copy
import random
import statistics
import numpy as np
import torch


def check_small_elements(matrix, threshold=1e-10):
    for row in matrix:
        for element in row:
            if abs(element) < threshold:
                return False
    return True

def get_label_distribution(dataloader):
    labels = []
    for _, batch_labels in dataloader:
    # Append the batch of labels to the list
        labels.extend(batch_labels.tolist())
    return labels
    
def get_data_size(dataloader):
    return len(get_label_distribution(dataloader))

def get_labels_variance(dataloader):
    return statistics.variance(get_label_distribution(dataloader))

def message_passing_round(G):
    for node in G.nodes():
        received_messages = {'models': []}  # Initialize an empty dictionary to store received messages
        # Update mailbox with relevant information
        for neighbor in G.neighbors(node):
            # Add messages from neighbors to the received_messages dictionary
            for key in received_messages.keys():
                received_messages[key].extend(G.nodes[neighbor]['mailbox'][key])
        
        # Update the node's mailbox with the new messages
        for key in received_messages.keys():
            for element in received_messages[key]:
                if element not in G.nodes[node]['mailbox'][key]:
                    G.nodes[node]['mailbox'][key].append(element)

def send_message_to_neighbor(G):
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            comm_charge = []
            for message in G.nodes[node]["mailbox"]:
                if neighbor not in message["tags"]:
                    comm_charge.append(1)
                    message["tags"].append(neighbor)
                    G.nodes[neighbor]["received_messages"].append(message)
            G.nodes[node]["comm_charge"][neighbor].append(sum(comm_charge))
            #print(f"COMM CHARGE FROM node {node} to {neighbor}: {sum(comm_charge)}")


def update_mailbox(G):
    for node in G.nodes():
        G.nodes[node]["mailbox"].extend(G.nodes[node]["received_messages"])
        G.nodes[node]["received_messages"] = []

def k_message_passing_round(G, k):
    for _ in range(k):
        send_message_to_neighbor(G)
        update_mailbox(G)

def weighted_aggregation(w, agg_weights, device):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0].to(device).state_dict())
    total_agg_weights = torch.tensor(sum(agg_weights), device=device)
    for key in w_avg:
        w_avg[key] *= (torch.tensor(agg_weights[0], device=device)/total_agg_weights)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i].to(device).state_dict()[key] * (torch.tensor(agg_weights[i], device=device)/total_agg_weights)
    return w_avg

def update_nodes(G, device):
    for node in G.nodes():
        #w = G.nodes[node]['mailbox']['models']
        w = []
        for message in G.nodes[node]['mailbox']:
            w.append(message["model"])

        #w = [message["model"] for message in G.nodes[node]['mailbox']]
        node_global_model = weighted_aggregation(w, [1] * len(w), device)
        #node_global_model = weighted_aggregation(w, get_num_clients_per_cluster(G))
        G.nodes[node]['model'].load_state_dict(node_global_model)
        G.nodes[node]['mailbox'] = []
        for client in G.nodes[node]['clients']:
            client.set_parameters(node_global_model)

def get_num_clients_per_cluster(G):
    rst = []
    for node in G.nodes():
        rst.append(len(G.nodes[node]['clients']))
    return rst

def generate_random_association(num_gateways, num_clients):
    # Ensure each gateway is assigned at least once
    association = list(range(num_gateways))
    
    # Assign remaining clients randomly
    association += [random.randint(0, num_gateways - 1) for _ in range(num_clients - num_gateways)]
    
    num_client_per_edge = []
    for edge in range(num_gateways):
        num_client_per_edge.append(association.count(edge))
    
    return num_client_per_edge

def generate_random_list(N, M): ## size of N and sum of elements equals to M
    remaining = -1
    while remaining < 0:
      random_list = [random.randint(5, M) for _ in range(N-1)]
      total = sum(random_list)
      remaining = M - total
    
    random_list.append(remaining)
    print(random_list)
    return random_list

def distribute_clients(trainloaders, valloaders, num_nodes, thresh , ratios ,method='equal'):
    trainloaders_node = []
    valloaders_node = []
    num_clients = len(trainloaders)
    thresholds = generate_clients_thresholds(num_clients, thresh, ratios)
    thresholds_ = []
    if method == 'equal':
        step = num_clients // num_nodes
        for i in range(0, num_clients, step):
            if i + step > num_clients:
                trainloaders_node.append(trainloaders_node[i:])
                valloaders_node.append(valloaders_node[i:])
                thresholds_.append(thresholds[i:])
            else:
                trainloaders_node.append(trainloaders[i:i+step])
                valloaders_node.append(valloaders[i:i+step])
                thresholds_.append(thresholds[i:i+step])

        return trainloaders_node, valloaders_node , thresholds_
    elif method == 'random':
        steps = generate_random_association(num_nodes, num_clients)
        i = 0
        for step in steps:
            if step != 0:
                if i + step > num_clients:
                    trainloaders_node.append(trainloaders[i:])
                    valloaders_node.append(valloaders[i:])
                    thresholds_.append(thresholds[i:])
                else:
                    trainloaders_node.append(trainloaders[i:i+step])
                    valloaders_node.append(valloaders[i:i+step])
                    thresholds_.append(thresholds[i:i+step])
            else:
                trainloaders_node.append([])
                valloaders_node.append([])
                thresholds_.append([])
            i = step

        return trainloaders_node, valloaders_node, thresholds_
    
def generate_clients_thresholds(N, values = [0.3, 0.6, 0.9], ratios = [0.2, 0.2, 0.6]):
    shuffled_values = []

    for value, ratio in zip(values, ratios):
        num_values = int(N * ratio)
        shuffled_values.extend([value] * num_values)

    random.shuffle(shuffled_values)

    return shuffled_values


def calculate_mean_quartiles(lst):
    data = np.array(copy.deepcopy(lst))
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    between_q1_q3 = (data >= q1) & (data <= q3)

    values_between_q1_q3 = data[between_q1_q3]
    return np.mean(values_between_q1_q3)



def estimate_next_round_time(G, agg_type='iqm'):
    if agg_type =='mean':
        estimated_times = []
        for message in G.nodes[0]['mailbox']:
            estimated_times.extend(message["estimated_time"])
        return np.mean(estimated_times)
       
    elif agg_type == 'iqm':
        estimated_times = []
        for message in G.nodes[0]['mailbox']:
            estimated_times.extend(message["estimated_time"])
        return calculate_mean_quartiles(estimated_times), max(estimated_times)
    else:
        estimated_times = []
        for message in G.nodes[0]['mailbox']:
            estimated_times.extend(message["estimated_time"])
        return np.median(estimated_times)
    
def calculate_counts(G, estimator):
    print("Estimated Max Time Between all Clients ", max(G.nodes[0]['mailbox']['estimated_time']))
    return np.sum(G.nodes[0]['mailbox']['estimated_time'] < estimator)
    
def analyze_dataloaders(dataloaders):
    label_distribution_per_client = []
    variance_per_client = []
    num_samples_per_client = []
    j=0
    for dataloader in dataloaders:
        labels = []
        for _, batch_labels in dataloader:
        # Append the batch of labels to the list
            labels.extend(batch_labels.tolist())
        label_distribution_per_client.append(labels)
        variance_per_client.append(statistics.variance(labels))
        num_samples_per_client.append(len(labels))
        print("Label Category in Client " , j,':',set(labels))
        print("Variance of labels in Client ", j , ':', variance_per_client[j])
        print("Number of Samples in Client ", j , ':', num_samples_per_client[j])
        print("*********", flush=True)
        j+=1


def get_comm_charge(G):
    comm_charges = []
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            comm_charges.append((node, neighbor, sum(G.nodes[node]["comm_charge"][neighbor])))
    return comm_charges



def get_model_size_in_mb(model):
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 ** 2)
        


def make_json_serializable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, dict):
        return {k: make_json_serializable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [make_json_serializable(v) for v in o]
    return o










