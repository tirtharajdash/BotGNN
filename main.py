import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks5 import  Net  		# import your network here
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='BOTDS',
                    help='dataset sub-directory under dir: data. e.g. BOTDS')
parser.add_argument('--epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--val_epochs', type=int, default=25,
                    help='maximum number of validation epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='type of pooling layer')
parser.add_argument('--use_node_attr', type=bool, default=True,
                    help='node features')


#Load GPU (If present)
args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    args.device = 'cuda:0'
    torch.cuda.manual_seed(args.seed)
    #the following two lines are optional (can be removed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Read dataset using TUDataset
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,use_node_attr=args.use_node_attr)
print(dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args.num_features)


#The following 4 lines for random train_test split (if you don't have explicit train/test split info)
#num_training = int(len(dataset)*0.6)
#num_val = int(len(dataset)*0.1)
#num_test = len(dataset) - (num_training+num_val)
#training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])


#Following 11 lines replaces the above 4 lines for this work (we have explicit train/test split info)
num_lines = sum(1 for line in open('./data/BOTDS/train_split'))
num_trn_lines = round(0.90*num_lines)
train_ids = np.loadtxt('./data/BOTDS/train_split', max_rows=num_trn_lines, dtype=int) - 1
val_ids = np.loadtxt('./data/BOTDS/train_split', skiprows=num_trn_lines, dtype=int) - 1
test_ids = np.loadtxt('./data/BOTDS/test_split', dtype=int) - 1
train_ids = train_ids.tolist()
val_ids = val_ids.tolist()
test_ids = test_ids.tolist()
training_set = Subset(dataset,train_ids)
validation_set = Subset(dataset,val_ids)
test_set = Subset(dataset,test_ids)


#Prepare the train, val and test sets using DataLoader
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
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
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


#Grid Search for Learning nhid of the network (Structure Learning with Validation)
nhid_grid = [8, 128]
best_loss = 1e10
print("Structure Selection:")
for nhid in nhid_grid:
    args.nhid = nhid
    model = Net(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for valepoch in range(args.val_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            #print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < best_loss:
        nhid_opt = nhid
        best_loss = val_loss


#Set nhid to Optimal value learned and build the model
args.nhid = nhid_opt
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("Model Created With Optimal Structure (nhid: {})".format(args.nhid))


#training of the selected model after nhid is learned using validation
min_loss = 1e10
patience = 0
print("Training Starts.")
for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        #print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 


#Evaluate the model on the Independent Test Set
model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc,test_loss = test(model,test_loader)
print("Independent Test Accuracy:{}".format(test_acc))


#store the results
with open('score.txt','w') as f:
    f.write('%d\t%f\t%f\n' % (args.nhid,test_loss,test_acc))
