import copy
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from cfg_parser import  get_dataset_and_model
from dataset import partition_and_prepare_dataset
from trainer import  evaluate_all_clients, evaluate_model, train_nodes
from network_creator import NetworkGraph, spawn_clients
from utils import analyze_dataloaders, distribute_clients, estimate_next_round_time, get_comm_charge, get_model_size_in_mb, k_message_passing_round, make_json_serializable, update_nodes
from torch.utils.data import ConcatDataset, DataLoader
import os
import torch.multiprocessing as mp





@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    global exp_name 
    exp_name = cfg.dataset + '_' + str(cfg.num_clients) + '_' + str(cfg.num_nodes) + '_' + str(cfg.dir_alpha)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    ## 1. Parse config
    print(OmegaConf.to_yaml(cfg))
    task = cfg.task
    ## 2. Construct the Network
    network_graph = NetworkGraph(cfg.num_nodes, "ring")
    network_graph.create_graph()
    
    ## 3. Determine number of message passing needed
    network_graph.calculate_num_message_passing()
    print(f"Number of message passing rounds : {network_graph.num_message_passing}")
    ## 4. Prepare Model and Dataset
    init_model, trainset, testset = get_dataset_and_model(cfg)

    ## 5. Partition the dataset
    trainloaders, validationloaders, testloader  = partition_and_prepare_dataset(trainset, 
                                                                                     testset ,
                                                                                     cfg.num_clients,
                                                                                     cfg.train_config['batch_size'] ,
                                                                                     csv = True,
                                                                                     partition=cfg.partition_method, 
                                                                                     dir_alpha= cfg.dir_alpha,
                                                                                     major_classes_num = 0,
                                                                                     val_ratio=0.1) # parition = ["noniid-#label", "noniid-labeldir", "unbalance", "iid"]
    
    ## 6. Prepare Clients
    analyze_dataloaders(trainloaders)
    thresh = cfg.client_category_thresholds
    ratios = cfg.client_category_ratios
    trainloaders_node , validationloaders_node, thresholds = distribute_clients(trainloaders, validationloaders, cfg.num_nodes, thresh, ratios ,method='random')
    clients_per_cluster = spawn_clients(trainloaders_node, validationloaders_node ,thresholds ,init_model,cfg)

    concatenated_dataset = ConcatDataset([loader.dataset for loader in trainloaders])
    concatenated_loader = DataLoader(concatenated_dataset, batch_size=cfg.train_config['batch_size'], shuffle=True)
    
    print("Model Num Parameters " , sum(p.numel() for p in init_model.parameters() if p.requires_grad))

    ## 7. Distribute datasets and global model on clients (nodes)
    for node in network_graph.graph.nodes():
        network_graph.graph.nodes[node]['model'] = copy.deepcopy(init_model)
        network_graph.graph.nodes[node]['estimated°time'] = []
        network_graph.graph.nodes[node]['clients'] = clients_per_cluster[node]
        network_graph.graph.nodes[node]['mailbox'] = []
        network_graph.graph.nodes[node]['comm_charge'] = {}

        for neighbor in network_graph.graph.neighbors(node):
            network_graph.graph.nodes[node]['comm_charge'][neighbor] = []


    round_time = float('inf')
    round_times = []
    beta = cfg.train_config['beta']
    if task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        ValueError("Invalid task_type. Please specify either 'binary' or 'multiclass'.")


    results = []

    exp_path = "outputs/"+ exp_name + "_" + str(os.getpid())
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    os.chdir(exp_path)
    print(f"EXPERIMENT SAVED IN {exp_path}")

    for round in range(cfg.num_rounds):
        ## 8. Local Training
        fl_round = round +1
        print(f'\n | Global Training Round : {fl_round} |\n', flush=True)

        train_nodes(network_graph , criterion  ,round_time ,fl_round, beta , task, device)

        # 9. Message Passing
        k_message_passing_round(network_graph.graph, network_graph.num_message_passing)
        # 10. Calculate Next Round Time
        round_time, max_time = estimate_next_round_time(network_graph.graph, 'iqm')
        round_times.append(round_time)

        #max_time =  max(network_graph.graph.nodes[0]['mailbox'][0]['estimated_time'])
        print("Estimated Max Time Between all Clients ", max_time)
        print("Next Round Calculated Time ", round_time)
        # 11. Update and Aggregate models
        update_nodes(network_graph.graph, device)

        # 12. Print performance on testloader
        accuracy_test, precision_test, recall_test, f1_test, total_loss_test, cm_test = evaluate_model(network_graph.graph.nodes[0]['model'], testloader, device, task, criterion)
        print("\n TEST SET METRICS: Accuracy: ", accuracy_test, " Precision: ", precision_test, " Recall: ", recall_test, " F1 score: ", f1_test, ' Total Loss: ', total_loss_test, ' Confusion Matrix:', cm_test)
        # 13. Print performance on trainloader
        accuracy, precision, recall, f1, total_loss, cm_train = evaluate_model(network_graph.graph.nodes[0]['model'], concatenated_loader, device, task, criterion)
        print("\n TRAINING SET METRICS: Accuracy: ", accuracy, " Precision: ", precision, " Recall: ", recall, " F1 score: ", f1, ' Total Loss: ', total_loss, ' Confusion matrix:', cm_train)
        # 14. Print performance on clients validationloaders
        print("\n CLIENTS VALIDATION SET METRICS: ")
        clients_per_node = [network_graph.graph.nodes[node]["clients"] for node in network_graph.graph.nodes()]
        clients_performances = evaluate_all_clients(clients_per_node, device, criterion, task)

        if fl_round == 1:
            comm_charge = get_comm_charge(network_graph.graph)
            model_size = get_model_size_in_mb(init_model)

        # 14. Print performance on clients validationloaders
        results_dict = {"test_set":
                            {"Accuracy": accuracy_test,
                            "Precision": precision_test,
                            "Recall": recall_test,
                            "F1": f1_test,
                            "Loss": total_loss_test,
                            "Confusion_matrix": cm_test
                            },
                        "clients": clients_performances,
                        "max_time": max_time,
                        "estimated_next_round_time": round_time,
                        "comm_charge": comm_charge,
                        "model_size": model_size,
                        }
        results.append(results_dict)

    results_serializable = make_json_serializable(results)
    with open(exp_name + ".json", "w") as file:
        json.dump(results_serializable, file, indent=4)
    
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()