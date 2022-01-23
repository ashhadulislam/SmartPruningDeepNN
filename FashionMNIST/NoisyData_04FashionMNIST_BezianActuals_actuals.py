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
import os
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

def combineMasks(mask1,mask2):
    mask_new=[]
    for i in range(len(mask1)):
        m_1=mask1[i]
        m_2=mask2[i]
#         print(m_1[0])
#         print(m_2[0])        
        m_new=np.multiply(m_1,m_2)
#         print(m_new[0])        
        mask_new.append(m_new)
    return mask_new



def get_mask_compression(mask_whole_model):
    num_total=0
    num_non_zeros=0
    for mask_each_layer in mask_whole_model:
        num_total+=torch.numel(mask_each_layer)
        num_non_zeros+=torch.count_nonzero(mask_each_layer)
        
    return (num_total-num_non_zeros)/num_total

    
    
def prune_model_get_mask_bezier(model,range_params,upscale,upscale_method):
    ax,bx,cx,dx=range_params
    mask_whole_model=[]
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
            mask_layer=torch.ones(params.shape)
#             print(nm,params.shape)
            num_components=params.shape[0]
#             print("Number of components are ",num_components)
            number_of_bins=int(math.sqrt(torch.numel(params[0])))
            if number_of_bins%2==1:
                number_of_bins+=1
#             print("Number of bins = ",number_of_bins)
            # creating the bezier curve
            x=np.linspace(1,0,number_of_bins//2)
            y1=ax*(1-x)**3 + bx*3*x*(1-x)**2 + cx*3*(x**2)*(1-x) + dx*(x**3)
            x=np.linspace(0,1,number_of_bins//2)
            y2=ax*(1-x)**3 + bx*3*x*(1-x)**2 + cx*3*(x**2)*(1-x) + dx*(x**3)

            proportion=torch.tensor(np.concatenate((y1,y2)))
            # normalizing to be between 0 and 1
            proportion=(proportion-torch.min(proportion))/(torch.max(proportion)-torch.min(proportion))
            # now let us scale up the values
            if upscale_method=="+":
                proportion=proportion+upscale
            elif upscale_method=="*":
                proportion=proportion*upscale            
#             print("Proportion is ",proportion)



            for index_component in range(num_components):
#                 print("Component number ",index_component)
                values=params[index_component]
    #             print(values.shape)
    #             print("number of non 0s",torch.count_nonzero(values))
                if len(values.shape)==3:
                    re_shaped_values=values.flatten()
                else:
                    re_shaped_values=values
                mask_vals=torch.ones(re_shaped_values.shape)
    #             print("Shape of mask is ",mask_vals.shape)
                sorted_indices=torch.argsort(re_shaped_values)
    #             print("Sorted indices are ",sorted_indices)
                #sorts from lowest to highest
                min_weight=torch.min(re_shaped_values).item()
                max_weight=torch.max(re_shaped_values).item()            
                bin_vals=torch.histc(re_shaped_values, bins=number_of_bins, min=min_weight, max=min_weight)
    #             print("bins created for hist=",bin_vals.shape,"vals are ",bin_vals)
                start_index=0
                for bin_index  in range(bin_vals.shape[0]):
    #                 print("Bin number ",bin_index)
                    # find how many elements are there in each bin
                    each_bin_count=int(bin_vals[bin_index].item())
    #                 print("from",start_index,"to",start_index+each_bin_count)
                    elem_indices=sorted_indices[start_index:start_index+each_bin_count]
                    # find the proportion of elements to be pruned from this bin
                    proportion_to_prune=min(proportion[bin_index],1)
                    count_to_prune=int(len(elem_indices)*proportion_to_prune)
    #                 print("proportion and count to prune",proportion_to_prune,count_to_prune)
                    selected_indices_prune=random.sample(list(elem_indices),count_to_prune)
    #                 print(selected_indices_prune,"out of ",elem_indices,"to be pruned")
                    for to_p_index in selected_indices_prune:
                        mask_vals[to_p_index]=0                
                    start_index+=each_bin_count
                    # done with each bins

                # done with each neurons/filters
                mask_vals=mask_vals.reshape(values.shape)
                mask_layer[index_component]=mask_vals
    #             print("At the end of each component, ",values.shape,mask_vals.shape)
            # done with weight in paramter name
            mask_whole_model.append(mask_layer)

        # done with a layer
    
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
    


                
import math


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
        weight_description=None,mask_whole_model=None,range_params=None,upscale=None,
       model_state_path=None,upscale_method=None):
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
        # Validation phase
        if mask_whole_model and range_params and upscale:
            # regenerate mask
            new_mask_list=prune_model_get_mask_bezier(model,range_params,upscale,upscale_method)
            mask_whole_model=combineMasks(new_mask_list,mask_whole_model)
            apply_mask_model(model,mask_whole_model)

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
    return history, weight_description,mask_whole_model


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


print("Program FashionMNIST 04_Bezier_Actual_Weights.py")

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
training_data=torch.tensor(train_ds.dataset.data)

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
model.load_state_dict(torch.load(model_state_path))


result = evaluate(model, test_loader)
print("Test result is ",result)


total_size,nz_size=model_size(model)
compression=(total_size-nz_size)/total_size
print("Compression un loaded=",compression)




range_params_list=[
#     [1, 0.14, 0.14, 0.1],
    [1, 0.83, 0.83, 0.5],    
    [1, 0.85, 0.85, 0.4],        
    [1, 0.81, 0.81, 0.6],        
#     [0.9, 0.9, 0.9, 0.5]
    
]

upscale_list=[
     0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,
#     0.4,0.41,0.43,0.47,0.49,
    
    0.5,
    0.55,    
    0.6,
    0.65,    
    0.7,
    0.75,    
    0.8,
    0.85,
    0.9,

]       
upscale_method_list=["*"]


target_compression=0.85
print("Running NoisyData_04FashionMNIST_BezianActuals_actuals.py")

for upscale_method in upscale_method_list:
    for range_params in range_params_list:
        for upscale in upscale_list:
            
            # the mask part
            pth_persist_mask=os.path.join("pickles","masks","04_bez_on_actuals",str(range_params))
            if not os.path.exists(pth_persist_mask):
                os.mkdir(pth_persist_mask)
            
            pth_persist_mask=os.path.join(pth_persist_mask,str(upscale))
            if not os.path.exists(pth_persist_mask):
                os.mkdir(pth_persist_mask)
            
            pth_persist_mask=os.path.join(pth_persist_mask,str(upscale_method))
            if not os.path.exists(pth_persist_mask):
                os.mkdir(pth_persist_mask)
                
            # the model part
            pth_persist_mask=os.path.join("model_state","04_bez_on_actuals",str(range_params))
            if not os.path.exists(pth_persist_mask):
                os.mkdir(pth_persist_mask)
            
            pth_persist_mask=os.path.join(pth_persist_mask,str(upscale))
            if not os.path.exists(pth_persist_mask):
                os.mkdir(pth_persist_mask)
            
            pth_persist_mask=os.path.join(pth_persist_mask,str(upscale_method))
            if not os.path.exists(pth_persist_mask):
                os.mkdir(pth_persist_mask)    
                
            model_state_path="model_state/mod.pt"




#             model_state_path="model_state/04_bez_on_actuals/[0.9995, 0.9995, 0.9995, 0.9994]/0.97/+/5799.pt"
#             mask_path="pickles/masks/04_bez_on_actuals/[0.9995, 0.9995, 0.9995, 0.9994]/0.97/+/5799.pkl"
        
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_state_path))
            else:
                model.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))

            list_mask_whole_model=prune_model_get_mask_bezier(model,
                                                              range_params,upscale,upscale_method)  
#             list_mask_whole_model=pickle.load( open( mask_path, "rb" ) )
            
    
            i=0
            turns=250
            while i<turns:
                this_turn_model_state_path="model_state/04_bez_on_actuals/"+str(range_params)+"/"+str(upscale) + "/" + str(upscale_method) + "/" + str(i) + ".pt"
                this_turn_list_mask_whole_model_path="pickles/masks/04_bez_on_actuals/"+str(range_params)+"/"+str(upscale) + "/" + str(upscale_method) + "/" + str(i) + ".pkl"
                if os.path.isfile(this_turn_model_state_path):
                    print(i,"Model exists, loading")
                    if torch.cuda.is_available():
                        model.load_state_dict(torch.load(this_turn_model_state_path))
                    else:
                        model.load_state_dict(torch.load(this_turn_model_state_path,
                                                         map_location=torch.device('cpu')))
                    list_mask_whole_model=pickle.load( open( this_turn_list_mask_whole_model_path, "rb" ) )
                    apply_mask_model(model,list_mask_whole_model)
                    
                else:                                                        
                    print(range_params,upscale,upscale_method,i,"/",turns)
                    # apply the mask on the model
                    epochs=1
                    history_prune,_,list_mask_whole_model = fit(epochs, lr, model, 
                                                                train_loader, val_loader,
                                      mask_whole_model=list_mask_whole_model,
                                             range_params=range_params,
                                             upscale=upscale,
                                            upscale_method=upscale_method)
                    torch.save(model.state_dict(), this_turn_model_state_path)
                    pickle.dump( list_mask_whole_model, 
                                open( this_turn_list_mask_whole_model_path, "wb" ) )
                    
                    total_size,nz_size=model_size(model)
                    compression=(total_size-nz_size)/total_size
                    res = evaluate(model, test_loader)
                    print("Compression=",compression,"Result after pruning is ",res,"Going for re training")
                    
                    if compression >=target_compression:
                        print("Target compression achieved, breaking")
                        break                    
                    
                i+=1
                print("*"*20)
            print("="*20)
                

            print("At the end of compression")
            total_size,nz_size=model_size(model)
            compression=(total_size-nz_size)/total_size
            res = evaluate(model, test_loader)
            print("Compression=",compression,"Result after re training is ",res,"going to check compression")                 
            if compression<target_compression:
                print("Still low compression")
                continue

    
    
            retrain_num=20
            print("GOing for re training for",retrain_num,"epochs")      
            for retrain_count in range(0,retrain_num):
                epochs=1
                new_upscale=0.0001
                history_prune,_,list_mask_whole_model = fit(epochs, lr, model, train_loader, val_loader,
                                            mask_whole_model=list_mask_whole_model,
                                            range_params=range_params,
                                            upscale=new_upscale,
                                            upscale_method=upscale_method)

                total_size,nz_size=model_size(model)
                compression=(total_size-nz_size)/total_size
                res = evaluate(model, test_loader)
                print("Retrain Number ##",retrain_count,"Compression=",compression,"Result after re training is ",res)




            print("At the end of retraining")
            total_size,nz_size=model_size(model)
            compression=(total_size-nz_size)/total_size
            res = evaluate(model, test_loader)
            print("Compression=",compression,"Result after re training is ",res,"Going for re training")                 




            print("GOing for noise calculation")      


            total_size,nz_size=model_size(model)
            compression=(total_size-nz_size)/total_size
            res = evaluate(model, test_loader)
            print(compression,res)

            # add gaussian noise

            class AddGaussianNoise(object):
                def __init__(self, mean=0., std=1.):
                    self.std = std
                    self.mean = mean

                def __call__(self, tensor):
                    return tensor + torch.randn(tensor.size()) * self.std + self.mean

                def __repr__(self):
                    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

            stdevs=np.linspace(0,1,30)

            results={}
            results["stdev"]=[]
            results["val_1"]=[]
            for stdev in stdevs:
                transform_noisy=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    AddGaussianNoise(0., stdev)
                ])
                # Define test dataset
                test_dataset = FashionMNIST(root='data/', train=False,transform=transform_noisy)
                test_loader = DataLoader(test_dataset, batch_size=256)
                test_loader = DeviceDataLoader(test_loader, device)

                res = evaluate(model, test_loader)

                print(stdev,res)
                results["stdev"].append(stdev)
                results["val_1"].append(res["val_acc"])    


            df_res=pd.DataFrame(results)
            file_stored="results_sheet/noisy/04BezActls" + str(compression) + "_" + str(range_params) + "_" + str(upscale_method) + "_" + str(upscale) +"re_"+str(retrain_num)+ ".csv"
            print("Result stored at ",file_stored)
            df_res.to_csv(file_stored,index=False)         
    