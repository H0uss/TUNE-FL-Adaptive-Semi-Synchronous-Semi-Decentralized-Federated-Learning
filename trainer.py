import copy
from statistics import mean
import numpy as np
import random
import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import get_data_size, weighted_aggregation
import torch.nn.functional as F
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed



warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def validate(net, criterion, validloader, device):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in validloader:
            #data, labels = data.to(device), labels.to(device)
            data = data.to(device)
            predict = net(data)
            num_classes = predict.shape[1]
            if num_classes > 2:
                labels = labels.to(device)
            else:
                labels = labels.view(-1, 1).to(device)
            loss = criterion(predict, labels)
            total_loss += loss.item()

    return total_loss / len(validloader)

def train(net, criterion, trainloader, optimizer, sleep_time, threshold, fl_round ,max_time, epsilon ,device):
    start_time = time.time()  # Record the start time
    i = 0
    best_val_loss = float('inf')
    net.to(device)
    while True:
        for data, labels in trainloader:
            i += 1
            net.train()
            #data, labels = data.to(device), labels.to(device)
            data = data.to(device)
            #print(data)
            optimizer.zero_grad()
            predict = net(data)
            num_classes = predict.shape[1]
            if num_classes > 2:
                labels = labels.to(device)
            else:
                labels = labels.view(-1, 1).to(device)
            
            loss = criterion(predict, labels)   
            loss.backward()
            optimizer.step()
            # Check if the elapsed time has reached the maximum time
            elapsed_time = time.time() - start_time
            if fl_round != 1:
                if elapsed_time >= max_time:
                    val_loss = validate(net, criterion, trainloader, device)
                    if abs(val_loss - best_val_loss) <=  epsilon:
                        return max_time, epsilon/abs(val_loss - best_val_loss), i
                    else:
                        return max_time, 1, i

            random_value = random.uniform(0, 1)
            # If the random value is above the threshold, pause the training
            if random_value > threshold:
                time.sleep(sleep_time)

        # Validate the model on the validation set
        val_loss = validate(net, criterion, trainloader, device)
        #print(val_loss)
        # Check for convergence
        if abs(val_loss - best_val_loss) <=  epsilon:
            end_time = time.time()
            elapsed_time = end_time - start_time
            return elapsed_time, 1, i

        best_val_loss = min(best_val_loss, val_loss)

    

def evaluate_model(model, dataloader, device, task_type, loss_fn):
    model.eval() 
    y_true = []
    y_pred_prob = []
    total_loss = 0
    model.to(device)
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            #inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            num_classes = outputs.shape[1]
            if num_classes > 2:
                labels = labels.to(device)
            else:
                labels = labels.view(-1, 1).to(device)
            #loss = loss_fn(outputs, labels.view(-1))
            loss = loss_fn(outputs, labels)
            if num_classes >2:
                outputs = outputs
                #outputs = torch.softmax(model(inputs), dim=1)
            else:
                outputs = torch.sigmoid(model(inputs))
            total_loss += loss.item()

            if task_type == "multiclass":
                _, y_pred = torch.max(outputs, 1)
                all_predictions.extend(y_pred.cpu().numpy())

            else:
                y_pred_prob.extend(outputs.cpu().numpy())

            y_true.extend(labels.cpu().numpy())
            

    if task_type == 'binary':
        y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]  # Convert probabilities to binary predictions
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        total_loss = total_loss/len(dataloader) if len(dataloader)!=0 else total_loss
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, precision, recall, f1, total_loss, cm
    
    elif task_type == 'multiclass':
        accuracy = accuracy_score(y_true, all_predictions)
        precision = precision_score(y_true, all_predictions, average='weighted', zero_division=1)
        recall = recall_score(y_true, all_predictions, average='weighted', zero_division=1)
        f1 = f1_score(y_true, all_predictions, average='weighted', zero_division=1)
        cm = confusion_matrix(y_true, all_predictions)
        total_loss = total_loss/len(dataloader) if len(dataloader)!=0 else total_loss

        return accuracy, precision, recall, f1, total_loss, cm
    else:
        raise ValueError("Invalid task_type. Please specify either 'binary' or 'multiclass'.")
    
    
def evaluate_random_clients(clients_per_cluster, device, loss_fn, task):
    for cluster in clients_per_cluster:
        b = True
        while b:
            try:
                client = random.choice(cluster)
                accuracy, precision, recall, f1, total_loss, b_accuracy, roc_auc = evaluate_model(client.model, client.valloader, device, task, loss_fn)
                b = False
                print(f"\n Client {client.id}: Accuracy: ", accuracy, " Precision: ", precision, " Recall: ", recall, " F1 score: ", f1, ' Total Loss: ', total_loss, " Balanced Accuracy: ",b_accuracy, " ROC_AUC: ", roc_auc)
            except:
                b = True

def evaluate_all_clients(clients_per_cluster, device, loss_fn, task):
    clients_performances = []
    for cluster in clients_per_cluster:
        for client in cluster:
            accuracy, precision, recall, f1, total_loss, cm = evaluate_model(client.model, client.valloader, device, task ,loss_fn)
            has_finished = client.finish_counter[-1]
            client_type = client.threshold
            print(f"\n Client {client.id}: Accuracy: ", accuracy, " Precision: ", precision, " Recall: ", recall, " F1 score: ", f1, ' Total Loss: ', total_loss, 
                  "Confusion matrix: ", cm ," Has finished: ", has_finished, " Client Type: ", client_type)
            clients_performances.append({"Id": client.id, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "loss": total_loss, 
                                         "Confusion_matrix":cm ,"finished": has_finished, "type":client_type, "agg_weight": client.agg_weight, "is_selected": client.is_selected})
    return clients_performances
   
                

def process_client(client, criterion ,fl_round, round_time, beta, device, task):
    client.is_selected = 0
    end_time, delay_weight, num_iterations = client.fit(criterion, fl_round, round_time,device)
    model = client.model
    accuracy_val, precision_val, recall_val, f1_val, total_loss_val, cm = evaluate_model(model, client.valloader, device, task, criterion)
    client.local_performance = {"Accuracy": accuracy_val, "Precision": precision_val, "Recall": recall_val, "F1": f1_val, "loss": total_loss_val, "Confusion matrix": cm}
    client.f1 = f1_val
    data_size = get_data_size(client.trainloader)
    client.data_size = data_size

    if fl_round == 1:
            client.set_round_time(end_time)
            client.agg_weight = data_size
            client.is_selected += 1

    else:
        #client.update_util_delta()
        client.agg_weight = data_size * delay_weight

        if end_time >= round_time:  # didn't finish
            client.set_round_time(beta *round_time + (1-beta) * client.next_round_time)
            client.finish_counter.append(0)
        else: # finished
            client.set_round_time((1-beta) *end_time  + beta* round_time)
            client.finish_counter.append(1)
    

    return model , client






def train_nodes(network_graph , criterion  ,round_time ,fl_round, beta , task, device):
    train_clients = []
    device = torch.device(device)
    for node in network_graph.graph.nodes():
        train_clients.extend(network_graph.graph.nodes[node]['clients'])
    
    finished_clients = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_client, client, criterion, fl_round, round_time, beta, device, task): client
            for client in train_clients
            }
            
        for future in as_completed(futures):
            _ , client = future.result()
            finished_clients.append((client, client.id))

    for node in network_graph.graph.nodes(): 
        estimated_times = []
        local_models = []
        clients_ids = []
        clients = []
        agg_weights = []

        for client in network_graph.graph.nodes[node]['clients']:
            agg_weights.append(client.agg_weight)
            clients.append(client)
            clients_ids.append(client.id)

            estimated_times.append(client.next_round_time)
            local_models.append(client.model)


        network_graph.graph.nodes[node]["received_messages"] = []
            

        if len(clients) == 1: # Only one client in the edge server
            network_graph.graph.nodes[node]['model'].load_state_dict(local_models[0].state_dict())
            network_graph.graph.nodes[node]['estimated_time'] = estimated_times
            message = {"model": network_graph.graph.nodes[node]['model'], "estimated_time": estimated_times, "tags": [node], "src": node}
            network_graph.graph.nodes[node]['mailbox'] = [message]
        
        elif fl_round != 1:
            increment_selection_counter(clients, clients_ids)
            print(f"the clients in node {node} are : {clients_ids};  the selected client ids are {clients_ids}")
            selected_agg_weights = [client.agg_weight for client in clients if client.id in clients_ids]
            selected_models = [client.model for client in clients if client.id in clients_ids]
          
            network_graph.graph.nodes[node]['model'].load_state_dict(weighted_aggregation(selected_models, selected_agg_weights, device))
            network_graph.graph.nodes[node]['estimated_time'] = estimated_times
            message = {"model": network_graph.graph.nodes[node]['model'], "estimated_time": estimated_times, "tags": [node], "src": node}
            network_graph.graph.nodes[node]['mailbox'] = [message]

        else:
            update_agg_weights1st_fl_round(network_graph,node, agg_weights)
            network_graph.graph.nodes[node]['model'].load_state_dict(weighted_aggregation(local_models, agg_weights, device))
            network_graph.graph.nodes[node]['estimated_time'] = estimated_times
            message = {"model": network_graph.graph.nodes[node]['model'], "estimated_time": estimated_times, "tags": [node], "src": node}
            network_graph.graph.nodes[node]['mailbox'] = [message]



def update_agg_weights1st_fl_round(network_graph, node, weights):
    for client in network_graph.graph.nodes[node]['clients']:
        client.agg_weight = client.agg_weight / (sum(weights))

def increment_selection_counter(clients, clients_ids):
    for client in clients:
        if client.id in clients_ids:
            client.selection_counter +=1
            client.is_selected = 1


def sum_models(w1, w2, beta):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w1.state_dict())
    for key in w_avg.keys():
        w_avg[key] += w2[key] * beta
    return w_avg


def iqm(data):
    # Sort the data
    sorted_data = sorted(data)
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    iq_values = [x for x in sorted_data if q1 <= x <= q3]
    # Calculate the mean of the interquartile values
    return np.mean(iq_values)






    



