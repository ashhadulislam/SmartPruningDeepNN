import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import pickle

from scipy import stats
    

import math
import random
import pandas as pd


def countDiffMasks(mask1,mask2):
    total_diff=0
    for i in range(len(mask1)):
        m_1=mask1[i].flatten()
        m_2=mask2[i].flatten()
        count_same=(m_1 == m_2).sum()
        count_different=m_1.flatten().shape[0]-count_same
        total_diff+=count_different
    return total_diff


def get_mask_compression(mask_whole_model):
    num_total=0
    num_non_zeros=0
    for mask_each_layer in mask_whole_model:
        num_total+=torch.numel(mask_each_layer)
        num_non_zeros+=torch.count_nonzero(mask_each_layer)
        
    return (num_total-num_non_zeros)/num_total

    
    

def prune_model_get_mask(model,prune_rate):
    '''
    works purely on the model to get
    mask
    '''
    mask_whole_model=[]
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            mask_layer=torch.ones(params.shape)
#             print(nm,params.shape)
            abs_var=torch.std(torch.abs(params.data))
#             print(abs_var)
#             print(params)
            threshold=abs_var*prune_rate
            num_components=params.shape[0]
            for index_component in range(num_components):
                values=params[index_component]            
                re_shaped_values=values.flatten()                
                mask_vals = (torch.abs(re_shaped_values)>threshold).float()
                mask_vals=mask_vals.reshape(values.shape)
#                 print(mask_vals.shape)
                mask_layer[index_component]=mask_vals
            mask_whole_model.append(mask_layer)
    return mask_whole_model
 
    
def get_thresholds_each_layer(model,prune_rate):
    thresholds_per_layer=[]
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            mask_layer=torch.ones(params.shape)
            abs_std=torch.std(torch.abs(params.data))
            threshold=abs_std*prune_rate
            thresholds_per_layer.append(threshold)
    return thresholds_per_layer
    
                
def apply_mask_model(model,list_mask_whole_model):
    mask_layer_count=0
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            mask_layer=list_mask_whole_model[mask_layer_count]
            with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                 print("Devices are ",params.device,mask_layer.device)
                mask_layer=mask_layer.to(device)
    
                params.data=params.data*mask_layer            
            mask_layer_count+=1
    

def store_weights_in_dic(weight_description,model):
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            if nm not in weight_description:
                weight_description[nm]={}
            num_components=params.shape[0]
            for index_component in range(num_components):
                if index_component not in weight_description[nm]:
                    weight_description[nm][index_component]={}
                values=params[index_component]
                flat_values=values.flatten()
                for index_wt in range(flat_values.shape[0]):
                    if index_wt not in weight_description[nm][index_component]:
                        weight_description[nm][index_component][index_wt]=[]
                    weight_description[nm][index_component][index_wt].append(flat_values[index_wt].detach().item())
    return weight_description


def get_boolean_dict_weight_dict(weight_description,prune_rate,thresholds_per_layer,last_few):
    '''
    works on the dictionary of weights
    to create a dict of 1s and 0s to show
    how many times weight is more than threshold
    per layer
    '''
    boolean_weight_description={}
    count=0
    for layer in weight_description.keys():  
#         print("Count = ",count)
        threshold_this_layer=thresholds_per_layer[count]
#         print("Threshold for layer ",count,layer,"is ",threshold_this_layer)
        if layer not in boolean_weight_description:
            boolean_weight_description[layer]={}
        for index_component in weight_description[layer].keys():
            if index_component not in boolean_weight_description[layer]:
                boolean_weight_description[layer][index_component]={}
            for index_wt in weight_description[layer][index_component].keys():
                if index_wt not in boolean_weight_description[layer][index_component]:
                    boolean_weight_description[layer][index_component][index_wt]=[]
                all_wts=weight_description[layer][index_component][index_wt][:-last_few]
                all_wts_boolean=[]
                for wt in all_wts:
                    if abs(wt)>threshold_this_layer:
                        all_wts_boolean.append(1)
                    else:
                        all_wts_boolean.append(0)
                boolean_weight_description[layer][index_component][index_wt]=all_wts_boolean                    
        count+=1
        
    return boolean_weight_description



def get_mean_dict_weight_dict(weight_description,last_few):
    '''
    works on the dictionary of weights
    to create a dict of mean of weights simply
    for the last few epochs
    '''
    mean_weight_description={}

    for layer in weight_description.keys():  
        if layer not in mean_weight_description:
            mean_weight_description[layer]={}
        for index_component in weight_description[layer].keys():
            if index_component not in mean_weight_description[layer]:
                mean_weight_description[layer][index_component]={}
            for index_wt in weight_description[layer][index_component].keys():
                if index_wt not in mean_weight_description[layer][index_component]:
                    mean_weight_description[layer][index_component][index_wt]=[]
                all_wts=weight_description[layer][index_component][index_wt][-last_few:]                
                all_wts_mean=np.mean(all_wts)                
                mean_weight_description[layer][index_component][index_wt]=all_wts_mean                    
        
    return mean_weight_description



    
# create mask from boolean weight dictionary
def create_mask_from_boolean_wt(model,boolean_wt_dict):
    mask_whole_model=[]
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            mask_layer=torch.ones(params.shape)
#             print(nm,params.shape)
            abs_var=torch.var(torch.abs(params.data))
#             print(abs_var)
#             print(params)
#             threshold=abs_var*prune_rate
            num_components=params.shape[0]
            for index_component in range(num_components):
                values=params[index_component]            
                re_shaped_values=values.flatten() 
                mask_vals=[]
                for val_index in range(re_shaped_values.shape[0]):
                    boolean_vals=boolean_wt_dict[nm][index_component][val_index]
                    m = stats.mode(boolean_vals)
#                     print("Verdict for this weight is ",m[0][0])
                    mask_vals.append(m[0][0])
#                 mask_vals = (torch.abs(re_shaped_values)>threshold).float()                
                mask_vals=np.asarray(mask_vals)
                mask_vals=mask_vals.reshape(values.shape)
#                 print(mask_vals.shape)
                mask_layer[index_component]=torch.from_numpy(mask_vals)
            mask_whole_model.append(mask_layer)
    return mask_whole_model


         
def create_mask_from_mean_wt(model,mean_weight_description,prune_rate):
    mask_whole_model=[]
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            mask_layer=torch.ones(params.shape)    
            # get all mean weights for this layer
            mean_wt_layer=mean_weight_description[nm]
            wts_this_layer=[]
            for neuron_index in list(mean_wt_layer.keys()):
                all_wts_this_neu=mean_wt_layer[neuron_index]

                for weight_index in list(mean_wt_layer[neuron_index].keys()):
                    wts_this_layer.append(mean_wt_layer[neuron_index][weight_index])
            # end of all neurons
#             print("first 5 Mean weights this layer are ",wts_this_layer[:5])
            abs_var=torch.std(torch.FloatTensor(wts_this_layer))
#             print("standard dev is ",abs_var)
            threshold=abs_var*prune_rate
#             print("Threshold is ",threshold)
            num_components=params.shape[0]            
            for index_component in range(num_components):
                values=params[index_component]            
                re_shaped_values=values.flatten()                
                mask_vals = (torch.abs(re_shaped_values)>threshold).float()
                mask_vals=mask_vals.reshape(values.shape)
#                 print(mask_vals.shape)
                mask_layer[index_component]=mask_vals
            mask_whole_model.append(mask_layer)
    return mask_whole_model
            
                
                
import math

def get_weighted_mean_dict_weight_dict(weight_description,last_few):
    '''
    works on the dictionary of weights
    to create a dict of mean of weights     
    for the last few epochs
    and gives a weightage to each weight value
    depending on the epoch
    last epoch highest weight
    '''
    weighted_mean_weight_description={}

    for layer in weight_description.keys():  
        if layer not in weighted_mean_weight_description:
            weighted_mean_weight_description[layer]={}
        for index_component in weight_description[layer].keys():
            if index_component not in weighted_mean_weight_description[layer]:
                weighted_mean_weight_description[layer][index_component]={}
            for index_wt in weight_description[layer][index_component].keys():
                if index_wt not in weighted_mean_weight_description[layer][index_component]:
                    weighted_mean_weight_description[layer][index_component][index_wt]=[]
                all_wts=weight_description[layer][index_component][index_wt][-last_few:]
                i_weights=[math.sqrt(i) for i in range(1,last_few+1)]
#                 print(all_wts,i_weights)
                all_wts_mean=np.average(all_wts,weights=i_weights)                
                weighted_mean_weight_description[layer][index_component][index_wt]=all_wts_mean                    
        
    return weighted_mean_weight_description    




# this part copied from shrinkbench

def nonzero(tensor):
    """Returns absolute number of values different from 0

    Arguments:
        tensor {numpy.ndarray} -- Array to compute over

    Returns:
        int -- Number of nonzero elements
    """
    return np.sum(tensor != 0.0)


def model_size(model, as_bits=False):
    """Returns absolute and nonzero model size

    Arguments:
        model {torch.nn.Module} -- Network to compute model size over

    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype

    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """

    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            t *= bits
            nz *= bits
        total_params += t
        nonzero_params += nz
    return int(total_params), int(nonzero_params)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))        

def correct(output, target, topk=(1,)):
    """Computes how many correct outputs with respect to targets

    Does NOT compute accuracy but just a raw amount of correct
    outputs given target labels. This is done for each value in
    topk. A value is considered correct if target is in the topk
    highest values of output.
    The values returned are upperbounded by the given batch size

    [description]

    Arguments:
        output {torch.Tensor} -- Output prediction of the model
        target {torch.Tensor} -- Target labels from data

    Keyword Arguments:
        topk {iterable} -- [Iterable of values of k to consider as correct] (default: {(1,)})

    Returns:
        List(int) -- Number of correct values for each topk
    """

    with torch.no_grad():
        maxk = max(topk)
        # Only need to do topk for highest k, reuse for the rest
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(torch.tensor(correct_k.item()))
        return res


# below copied from shrinkbench, to be used later
# def accuracy(model, dataloader, topk=(1,)):
#     """Compute accuracy of a model over a dataloader for various topk

#     Arguments:
#         model {torch.nn.Module} -- Network to evaluate
#         dataloader {torch.utils.data.DataLoader} -- Data to iterate over

#     Keyword Arguments:
#         topk {iterable} -- [Iterable of values of k to consider as correct] (default: {(1,)})

#     Returns:
#         List(float) -- List of accuracies for each topk
#     """

#     # Use same device as model
#     device = next(model.parameters()).device

#     accs = np.zeros(len(topk))
#     with torch.no_grad():

#         for i, (input, target) in enumerate(dataloader):
#             input = input.to(device)
#             target = target.to(device)
#             output = model(input)

#             accs += np.array(correct(output, target, topk))

#     # Normalize over data length
#     accs /= len(dataloader.dataset)

#     return accs





# LENET 300-100 for MNIST and comparison
class FashionMnistNet(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)        
        self.fc4 = nn.Linear(64, 10)
        self.fc4.is_classifier = True
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        top_1, top_5 = correct(out, labels,topk=(1,5))
#         print("Batch is ",batch[1].shape)
        
        top_1=top_1/batch[1].shape[0]
        top_5=top_5/batch[1].shape[0]

#         print("corr",top_1,top_5)
#         return {'val_loss': loss, 'val_acc': acc}
        return {'val_loss': loss, 'val_acc': acc, 'top_1': top_1, 'top_5': top_5}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        
        batch_top_1s = [x['top_1'] for x in outputs]
#         print(batch_top_1s)
        epoch_top_1 = torch.stack(batch_top_1s).mean()      # Combine top_1
        
        batch_top_5s = [x['top_5'] for x in outputs]
        epoch_top_5 = torch.stack(batch_top_5s).mean()      # Combine top_5
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(),
               'val_top_1': epoch_top_1.item(), 'val_top_5': epoch_top_5.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}, val_top_1: {:.4f}, val_top_5: {:.4f}".format(
                                epoch, result['val_loss'], result['val_acc'], 
                                result['val_top_1'], result['val_top_5']))
        
        
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
    
def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
#     print("outputs are ",outputs)
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD,
        weight_description=None,mask_whole_model=None,
              model_state_path=None):
    """Train the model using gradient descent"""
    print("At train")
    history = []
    best_so_far=-999            
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if mask_whole_model:
#                 print("Applying mask")
                apply_mask_model(model,mask_whole_model)
        # Validation phase
        result = evaluate(model, val_loader)
        if best_so_far<result["val_top_1"]:
            best_so_far=result["val_top_1"]
            if model_state_path:
                torch.save(model.state_dict(), model_state_path)
        
        model.epoch_end(epoch, result)
        history.append(result)
#         print("wt desc = ",weight_description)
        if weight_description!=None:
            print("going for weight")
            weight_description=store_weights_in_dic(weight_description,model)
    return history, weight_description


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


print("Program with weighht pruning, dynamic, many ranges")

print("Torch cuda ",torch.cuda.is_available())


device = get_default_device()
print("device ",device)



dataset = FashionMNIST(root='data/', download=True, transform=ToTensor())


# Define test dataset
test_dataset = FashionMNIST(root='data/', train=False,transform=ToTensor())

val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])


batch_size=128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256)

shape=dataset[0][0].shape
input_size=1
for s in shape:
    input_size*=s

    
    


num_classes = 10


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

targets=train_ds.dataset.targets
# training_data=torch.tensor(train_ds.dataset.data)
training_data=train_ds.dataset.data.clone().detach()

training_data = training_data.to(device=device)


model=FashionMnistNet()
if torch.cuda.is_available():
    model=model.cuda()

history = [evaluate(model, val_loader)]
print("initial result",history)
weight_description={}
epochs=50
lr=0.01



model_state_path="model_state/mod.pt"

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_state_path))
else:
    model.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))

weight_description= pickle.load( open( "pickles/weight_description.p", "rb" ) )

result = evaluate(model, test_loader)
print("Test result is ",result)


total_size,nz_size=model_size(model)
compression=(total_size-nz_size)/total_size
print("Compression=",compression)



last_few=50
mean_weight_description_weighted=get_weighted_mean_dict_weight_dict(weight_description,last_few)




metrics={}
metrics["prune_rate"]=[]
metrics["compression"]=[]
metrics["epochs"]=[]
metrics["top_5"]=[]
metrics["top_1"]=[]


df_base=pd.read_csv("results_sheet/02_py_mean_weighted.csv")
done_pruning_list=list(df_base["prune_rate"])




# prune_rate_range=[0.1,0.2,0.3,0.4,0.6,0.8,0.9,1.1,1.3,1.5,1.7,
#                   1.75,1.8,1.85,1.9,1.92,1.93,1.95,1.99,2.1,2.12,2.2,
#                  2.25,2.3,2.35,2.4,2.5,2.7,2.9,4]

prune_rate_range=[3.1,3.2,3.4,3.5,3.7,3.8]

for prune_rate in prune_rate_range:
    if prune_rate in done_pruning_list:
        continue

    model_state_path="model_state/mod.pt"
    model.load_state_dict(torch.load(model_state_path))
    mask_whole_model=create_mask_from_mean_wt(model,mean_weight_description_weighted,prune_rate)
    print("Mask compression = ",prune_rate,get_mask_compression(mask_whole_model))

    




    apply_mask_model(model,mask_whole_model)
    total_size,nz_size=model_size(model)
    compression=(total_size-nz_size)/total_size
    res = evaluate(model, test_loader)
    print("After masking, Compression=",compression,"Result after pruning is ",res)


    epochs=30
    pruned_model_state_path="model_state/threshold_mean_wts.pt"
    history_prune,_ = fit(epochs, lr, model, train_loader, val_loader,
                          mask_whole_model=mask_whole_model,
                         model_state_path=pruned_model_state_path)

    print("Pruning+Re training the weighted threshold way, complete")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(pruned_model_state_path))
    else:
        model.load_state_dict(torch.load(pruned_model_state_path,map_location=torch.device('cpu')))



    total_size,nz_size=model_size(model)
    compression=(total_size-nz_size)/total_size
    res = evaluate(model, test_loader)

    print("Compression=",compression,"Result after pruning is ",res)
    metrics["prune_rate"].append(prune_rate)
    metrics["compression"].append(compression)
    metrics["epochs"].append(epochs)
    metrics["top_5"].append(res['val_top_5'])
    metrics["top_1"].append(res['val_top_1'])

    

    
    

dataframe_results=pd.DataFrame(metrics)
print(dataframe_results.head())

dataframe_results=pd.concat([df_base, dataframe_results], ignore_index=True)


dataframe_results.to_csv("results_sheet/02_py_mean_weighted.csv",
                         index=False)
