import sys
from dataset import get_IoTIDS,  get_cicids, get_unsw_nb15, get_ciciomt , get_ciciomt_multi, get_IoTIDS_multi
from models import  CicIDS, CicIoMT_multi, CicIoMT, IoTIDSModel ,UnswModel, IoTIDSModel_multi

def get_dataset_and_model(cfg):
    dataset_name = cfg.dataset
    if dataset_name =="ciciomt":
        if cfg.task == "binary":
            model = CicIoMT()
            trainset, testset = get_ciciomt()
        elif cfg.task == "multiclass":
            model = CicIoMT_multi()
            trainset, testset = get_ciciomt_multi()
        else:
            raise ValueError("Tasks are: 'binary' or 'multiclass' ")
        
    elif dataset_name =="unsw":
        if cfg.task =="binary":
            model = UnswModel()
            trainset, testset = get_unsw_nb15()
        else:
            raise ValueError("UNSW-NB15 dataset only implemented for Binary task, please add the code for it")

    elif dataset_name =="cicids":
        if cfg.task == "binary":
            model = CicIDS()
            trainset, testset = get_cicids()
        else:
            raise ValueError("cicids dataset only implemented for Binary task, please add the code for it")
        
    elif dataset_name =="iotids":
        if cfg.task == "binary":
            model = IoTIDSModel()
            trainset, testset = get_IoTIDS()
        elif cfg.task == 'multiclass':
            model = IoTIDSModel_multi()
            trainset, testset = get_IoTIDS_multi()
        else:
            raise ValueError("Tasks are: 'binary' or 'multiclass' ")


    return model, trainset, testset



class DualOutput:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


        