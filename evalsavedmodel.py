import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks4 import Net  		# import your network here
import torch.nn.functional as F
import argparse
import os
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset



parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='BOTDS',
                    help='dataset sub-directory under dir: data. e.g. BOTDS')



#Load GPU (If present)
args = parser.parse_args()
args.device = 'cpu'
args.seed = 777
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'


#Read dataset using TUDataset
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,use_node_attr=True)
print(dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args.num_features)


#load the test split
test_ids = np.loadtxt('./data/BOTDS/train_split', dtype=int) - 1
test_ids = test_ids.tolist()
test_set = Subset(dataset,test_ids)


#Prepare the test set using DataLoader
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


#model testing (Independent Test Set)
def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        #print(pred.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


#get model prediction for a (batch of) data instance
def predict(model,data):
    model.eval()
    data = data.to(args.device)
    out = model(data)
    pred = out.max(dim=1)[1]
    with torch.no_grad():
        #print("True:{}, Predicted:{}".format(data.y.item(),pred.item()))
        return data.y.item(), pred.item()


#evaluation metrics
def get_eval_scores(Y_true, Y_pred):
    print("Confusion Matrix (Rows: True, Cols: Predicted):")
    print(confusion_matrix(Y_true,Y_pred))
    print("Precision: {}".format(precision_score(Y_true,Y_pred)))
    print("Recall: {}".format(recall_score(Y_true,Y_pred)))
    print("F1-score: {}".format(f1_score(Y_true,Y_pred)))
    print("Accuracy: {}".format(accuracy_score(Y_true,Y_pred)))


#load nhid and other model structure params
args.nhid = pd.read_csv("score.txt",sep="\t", header=None)[0][0]
args.pooling_ratio = 0.5
args.dropout_ratio = 0.5

#load the model (gpu)
model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))

#print model summary (uncomment the following line for this)
print(model)

#evaluate the model on the Independent Test Set
test_acc,test_loss = test(model,test_loader)
print("Test:: loss:{}\taccuracy:{}".format(test_loss,test_acc))

#individual predictions and evaluation metrics
print(len(test_loader))
Y = []
for data in test_loader:
    y_true, y_pred = predict(model,data)
    Y.append([y_true,y_pred])

Y = np.array(Y)
np.savetxt("preds.txt",Y,delimiter=',',fmt="%d")
get_eval_scores(Y[:,0],Y[:,1])
