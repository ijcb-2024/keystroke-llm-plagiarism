import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import calculate_eer
from torch.nn.functional import cosine_similarity
from torch.nn import CosineSimilarity
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm
import gc

class Trainer:
    def __init__(self, model, optimizer, device, loss, path, threshold=0.33):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        self.path = path
        self.threshold = threshold

    def train(self, train_loader, num_epochs, val_loader, test_loader):
        self.model.train()
        curr_epoch = 0
        cos = CosineSimilarity(dim=1)
        for epoch in range(curr_epoch,num_epochs+curr_epoch) :
            train_loss = 0
            val_loss = 0
            train_count = 0
            val_count = 0
            acc_train_avg = 0
            acc_val_avg = 0
            f1_train = 0
            f1_val = 0
            predicted_train = []
            label_train = []
            predicted_val = []
            label_val = []
            torch.cuda.empty_cache()
            
            eer_train = 0
            eer_val = 0
            eer_test = 0
            
            fpr_train = 0
            fpr_val = 0
            fpr_test = 0
            
            fnr_train = 0
            fnr_val = 0
            fnr_test = 0

            print("EPOCH - {}".format(epoch))
            
            ### Training Loop ###
            
            for item in tqdm(train_loader) :
                ## Train Model 
                try: 
                    self.model.train()
                    x1,x2,label,length1,length2 = item
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    label = label.to(self.device)
                    label = label.float()
                    self.optimizer.zero_grad()
                    output = self.model(x1,x2,length1,length2)
                    d = (cos(output[0],output[1]) + 1)/2
                    loss_ = self.loss(d,label)
                    eer_result = calculate_eer(d.detach().cpu().numpy(),np.array(label.cpu()))
                    eer_train += eer_result[0]/output[0].shape[0]    
                    fpr_train += eer_result[2]
                    fnr_train += eer_result[3]
                    
                    train_loss += loss_.item()
                    loss_.backward()
                    self.optimizer.step()

                    curr_pred = (d.detach().cpu().numpy() >= self.threshold)*1
                    curr_label = label.detach().cpu().numpy()
                    predicted_train.extend(curr_pred)
                    label_train.extend(curr_label)
                    
                    ## Accuracy Calculation
                    acc_curr = np.sum(curr_pred == curr_label)/output[0].shape[0]
                    acc_train_avg += acc_curr
                        
                    ## Increase Count
                    train_count += 1
                    
                    ## Delete variables
                    del output,x1,x2,length1,length2,label,loss_,curr_pred,curr_label,d,eer_result
                    
                except Exception as e:
                    continue

            with torch.no_grad() :
                try: 
                    for item in tqdm(val_loader) :
                    ## Testing Model 
                        x1,x2,label,length1,length2 = item
                        x1 = x1.to(self.device)
                        x2 = x2.to(self.device)
                        label = label.to(self.device)
                        label = label.float()
                        self.optimizer.zero_grad()
                        output = self.model(x1,x2,length1,length2)
                        d = (cosine_similarity(output[0],output[1]) + 1)/2
                        loss_ = self.loss(d,label)
                        val_loss += torch.mean(loss_).item()
                        
                        eer_result = calculate_eer(d.detach().cpu().numpy(),np.array(label.cpu()))
                        eer_val += eer_result[0]/output[0].shape[0]    
                        fpr_val += eer_result[2]
                        fnr_val += eer_result[3]    

                        curr_pred = (d.detach().cpu().numpy() >= self.threshold)*1
                        curr_label = label.detach().cpu().numpy()
                        predicted_val.extend(curr_pred)
                        label_val.extend(curr_label)
                        
                        ## Accuracy Calculation
                        acc_curr = np.sum(curr_pred == curr_label)/output[0].shape[0]
                        acc_val_avg += acc_curr
                        
                        ## Increase Count
                        val_count += 1

                        ## Deleting 
                        del output,x1,x2,length1,length2,label,loss_,curr_pred,curr_label,d,eer_result
                    
                except Exception as e:
                    continue

            with torch.no_grad() :
                try : 
                    for item in tqdm(test_loader) :
                        x1,x2,label,length1,length2 = item
                        x1 = x1.to(self.device)
                        x2 = x2.to(self.device)
                        label = label.to(self.device)
                        label = label.float()
                        self.optimizer.zero_grad()
                        self.model.eval()
                        output = self.model(x1,x2,length1,length2)
                        d = (cosine_similarity(output[0],output[1]) + 1)/2
                        loss_ = self.loss(d,label)
                        eer_result = calculate_eer(d.detach().cpu().numpy(),np.array(label.cpu()))
                        eer_test += eer_result[0]/output[0].shape[0]    
                        fpr_test += eer_result[2]
                        fnr_test += eer_result[3]
                    
                        test_count += 1
                        
                        
                        acc_avg_test += np.sum((d.detach().cpu().numpy() >= self.threshold)*1 == label.detach().cpu().numpy())/output[0].shape[0]
                        av_f1 += f1_score((d.detach().cpu().numpy() >= self.threshold)*1,label.detach().cpu().numpy())
                        av_prec += precision_score((d.detach().cpu().numpy() >= self.threshold)*1,label.detach().cpu().numpy())
                        av_recall += recall_score((d.detach().cpu().numpy() >= self.threshold)*1,label.detach().cpu().numpy())

                        del output,x1,x2,length1,length2,label,loss_,d,eer_result
                        
                except Exception as e:
                    print("Exception: ", e)
                    continue
            
            ## Saving Model Checkpoint
            if self.path: 
                torch.save(self.model,os.path.join(os.getcwd(),self.path,'_model{}.pth').format(epoch))
            # else: 
            #     path = new_path
            #     print(new_path)
            #     os.mkdir(os.path.join(os.getcwd(),self.path))
                
            ## Calculate f1 score
            f1_train = f1_score(label_train,predicted_train)
            prec_train = precision_score(label_train,predicted_train)
            prec_val = precision_score(label_val,predicted_val)
            recall_train = recall_score(label_train,predicted_train)
            recall_val = recall_score(label_val,predicted_val)
            f1_val = f1_score(label_val,predicted_val)
            
            print("Precision train/val/test: {}/{}/{}".format(prec_train,prec_val,av_prec/test_count))
            print("Recall train/val/test: {}/{}/{}".format(recall_train,recall_val,av_recall/test_count))
            print("F1 train/val/test: {}/{}/{}".format(f1_train,f1_val,av_f1/test_count))
            print("EER Train/Val/Test: {}/{}/{}".format(eer_train/train_count,eer_val/val_count))