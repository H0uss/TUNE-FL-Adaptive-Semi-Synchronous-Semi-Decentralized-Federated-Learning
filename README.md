# TUNE-FL-Adaptive-Semi-Synchronous-Semi-Decentralized-Federated-Learning


# Running the Experiments

## 1. Prepare the Data
Place the desired `train_set` and `test_set` files inside the `data/` directory.

## 2. Configure the Experiment
Edit the `base.yaml` file located in the `conf/` directory to set up your experiment parameters.

## 3. Adding a New Dataset
If you want to integrate a new dataset:
- Add the required dataset-loading functions to `dataset.py`
- Add the corresponding model-handling functions to `model.py`

## 4. Run the Experiment
Execute the following command:

```bash
python main.py --config-name=base


If you use this work, please cite the following paper:
@INPROCEEDINGS{10975902,
  author={Jmal, Houssem and Piamrat, Kandaraj and Aouedi, Ons},
  booktitle={2025 IEEE 22nd Consumer Communications & Networking Conference (CCNC)}, 
  title={TUNE-FL: Adaptive Semi-Synchronous Semi-Decentralized Federated Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Adaptive systems;Federated learning;Network topology;Image edge detection;Intrusion detection;Heterogeneous networks;Servers;Synchronization;Distributed computing;Semi-Decentralized federated learning;edge device heterogeneity;adaptive synchronization},
  doi={10.1109/CCNC54725.2025.10975902}}