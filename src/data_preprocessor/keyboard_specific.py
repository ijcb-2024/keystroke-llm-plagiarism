import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
import random

class CustomDataset(Dataset):
    def __init__(self,data,mask,label) :
        self.data = data
        self.mask = mask
        self.label = label

    def __getitem__(self,index) :
        x = torch.tensor(self.data[index],dtype=torch.float32)
        m = torch.tensor(self.mask[index],dtype=torch.float32)

        counts = np.unique(m[0,:],return_counts=True)
        length1 = counts[1][-1]
        counts = np.unique(m[1,:],return_counts=True)
        length2 = counts[1][-1]

        temp = torch.zeros(2)
        temp[self.label[index]] = 1

        return x[0,:],x[1,:],self.label[index],length1,length2

    def __len__(self) :
        return len(self.data)


class KeyboardSpecificPreprocessor:

    ##Key board map to map each key to a number
    keyboard_map = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
        'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
        '0': 27, 'd0' : 27, 'd1' : 28, '1': 28, '2': 29, 'd2': 29, '3': 30, 'd3': 30, '4': 31, 'd4' : 31,'5': 32, 'd5':32, '6': 33, 'd6':33, '7': 34, 'd7': 34, '8': 35, 'd8': 35, '9': 36, 'd9':36,
        'f1': 37, 'f2': 38, 'f3': 39, 'f4': 40, 'f5': 41, 'f6': 42, 'f7': 43, 'f8': 44, 'f9': 45, 'f10': 46,
        'f11': 47, 'f12': 48, 'esc': 49, '`': 50, '-': 51, 'subtract':51, '=': 52, 'backspace': 53,'back':53, 'tab': 54, '[': 55, ']': 56,
        '\\': 57, 'capslock': 58,'capital':58, ';': 59, '\'': 60, 'enter': 61, 'return':61, 'shift': 62, ',': 63, '.': 64, 'decimal':64, 'oemperiod': 64 , '/': 65 ,'divide':65, 'control': 66,'ctrl':66,
        'alt': 67, ' ': 68, 'printscreen': 69, 'scrolllock': 70,'scroll':70, 'pause': 71, 'insert': 72, 'home': 73, 'pageup': 74,
        'delete': 75, 'end': 76, 'pagedown': 77, 'arrowup': 78, 'arrowleft': 79, 'arrowdown': 80, 'arrowright': 81,
        'numlock': 82, 'numpad0': 83, 'numpad1': 84, 'numpad2': 85, 'numpad3': 86, 'numpad4': 87, 'numpad5': 88,
        'numpad6': 89, 'numpad7': 90, 'numpad8': 91, 'numpad9': 92, 'numpadmultiply': 93, 'numpadadd': 94,
        'numpadsubtract': 95, 'numpaddecimal': 96, 'numpaddivide': 97, 'numpadenter': 98, 'contextmenu': 99,
        'leftctrl': 100, 'leftshift': 101, 'leftshiftkey':101, 'leftalt': 102, 'lmenu':102, 'leftmeta': 103, 'rightctrl': 104,'rcontrolkey':104, 'rightshift': 105, 'rshiftkey': 105,
        'rightalt': 106, 'rmenu':106, 'rightmeta': 107, ':': 108, 'colon': 108, 'unidentified':0, ')': 109, '(':110, 'meta':111, '≠':112, '@':113, '>':114,'<':115, '*':116,'+':117,'add' : 117,'#':118,'$':119, '"':120, 'process':121,'_':122,
        '{':123,'}':124,'?':134,'f1':135,'f2':136,'f3':137,'f4':138,'f5':139,'f6':140,'f7':141,'f14':142,'´':143,'':0,'©':144,'escape':49,'clear':145,'lcontrolkey':100,'lshiftkey':101, 'space':68, 'left': 146, 'right': 147, 'up':148, 'down': 149, 'apps':150,
        'rwin' : 151, 'next': 152, 'lwin' : 153, 'browserback':154, 'browserforward':155, 'browserrefresh':156, 'browserstop':157, 'browsersearch':158, 'browserfavorites':159, 'browserhome':160, 'volumemute':161, 'volumedown':162, 'volumeup':163, 'medianexttrack':164, 'mediaprevioustrack':165
    }

    def __init__(self, seq_len, batch_size):
       self.M = seq_len
       self.batch_size = batch_size
    
    def get_train_test_sets(self, fixed_data_train, free_data_train, fixed_data_test, free_data_test):
        ## Converting data
        free_data_train = self.convert_data(free_data_train)
        fixed_data_train = self.convert_data(fixed_data_train)
        free_data_test = self.convert_data(free_data_test)
        fixed_data_test = self.convert_data(fixed_data_test)

        ## Create validation set
        free_data_val = {}
        fixed_data_val = {}

        for key in free_data_train.keys() :
            if len(free_data_train[key]) > 1 :
                val_sect = int(0.1*len(free_data_train[key]))
                free_data_val[key] = free_data_train[key][:val_sect]
                free_data_test[key] = free_data_train[key][val_sect:3*val_sect]
                free_data_train[key] = free_data_train[key][3*val_sect:]
                
                
        for key in fixed_data_train.keys() :
            if len(fixed_data_train[key]) > 1 :
                fixed_data_val[key] = fixed_data_train[key][:int(0.1*len(fixed_data_train[key]))]
                fixed_data_test[key] = fixed_data_train[key][int(0.1*len(fixed_data_train[key])):int(0.3*len(fixed_data_train[key]))]
                fixed_data_train[key] = fixed_data_train[key][int(0.3*len(fixed_data_train[key])):]

        ## Padding and clipping sequences

        mask_free_train = {}
        mask_fixed_train = {}
        mask_fixed_val = {}
        mask_free_val = {}
        mask_free_test = {}
        mask_fixed_test = {}

        free_data_train,mask_free_train = self.return_pad_seq(free_data_train)
        fixed_data_train,mask_fixed_train = self.return_pad_seq(fixed_data_train)
        fixed_data_val,mask_fixed_val = self.return_pad_seq(fixed_data_val)
        free_data_val,mask_free_val = self.return_pad_seq(free_data_val)
        free_data_test,mask_free_test = self.return_pad_seq(free_data_test)
        fixed_data_test,mask_fixed_test = self.return_pad_seq(fixed_data_test)

        # Combining Pairs

        new_data_train,y_data_train,mask_train = self.combine_pairs(fixed_data_train,free_data_train,mask_fixed_train,mask_free_train)
        new_data_val,y_data_val,mask_val = self.combine_pairs(fixed_data_val,free_data_val,mask_fixed_val,mask_free_val)
        new_data_test,y_data_test,mask_test = self.combine_pairs(fixed_data_test,free_data_test,mask_fixed_test,mask_free_test)

        combined_data_list_train,combined_mask_list_train,y_data_list_train = self.combine(new_data_train,mask_train,y_data_train)
        combined_data_list_val,combined_mask_list_val,y_data_list_val = self.combine(new_data_val,mask_val,y_data_val)
        combined_data_list_test,combined_mask_list_test,y_data_list_test = self.combine(new_data_test,mask_test,y_data_test)

        # Class Balancing for Training Set

        indexes = np.arange(len(combined_data_list_train))
        y_data_list_train = np.array(y_data_list_train)
        class_0_index = indexes[y_data_list_train == 0]
        class_1_index = indexes[y_data_list_train == 1]

        min_length = min(len(class_0_index),len(class_1_index))

        indexes = np.concatenate((class_0_index[:min_length],class_1_index[:min_length]))
        random.shuffle(indexes)

        combined_data_list_train = np.array(combined_data_list_train)
        combined_mask_list_train = np.array(combined_mask_list_train)
        y_data_list_train = np.array(y_data_list_train)

        combined_data_list_train = combined_data_list_train[indexes]
        combined_mask_list_train = combined_mask_list_train[indexes]
        y_data_list_train = y_data_list_train[indexes]

        del indexes,class_0_index,class_1_index
        gc.collect()

        train_data = CustomDataset(combined_data_list_train,combined_mask_list_train,y_data_list_train)
        val_data = CustomDataset(combined_data_list_val,combined_mask_list_val,y_data_list_val)
        test_data = CustomDataset(combined_data_list_test,combined_mask_list_test,y_data_list_test)

        train_loader = DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=True)
        test_loader = DataLoader(dataset=test_data,batch_size=self.batch_size,shuffle=True)

        return train_loader, val_loader, test_loader


    def divide_into_batches(self, x) :
        num = len(x)//self.M
        list_text = []

        for i in range(num):
            temp = x[int(i*len(x)/num):min(int((i+1)*len(x)/num),len(x))]
            
            if len(temp) < 0.8*self.M :
                continue
              
            list_text.append(temp)

        return list_text

    def process_data_buffalo(self, data, verbose=0) :
        for key in data.keys() :
            if verbose == 1 :
                data[key] = json.loads(data[key])
            
            timestamp_kd = []
            timestamp_ku = []
            list_ = data[key]["keyboard_data"]
            
            for i in range(len(data[key]["keyboard_data"])) :
                if list_[i][1].lower() == "kd" :
                    timestamp_kd.append(int(list_[i][2]) - int(list_[i-1][2]) if i > 0 else int(list_[i][2]))
                else :
                    timestamp_ku.append(int(list_[i][2]) - int(list_[i-1][2]) if i > 0 else int(list_[i][2]))
            
            timestamp_kd = np.array(timestamp_kd, dtype=np.float32)
            timestamp_ku = np.array(timestamp_ku, dtype=np.float32)
            
            if len(timestamp_kd) > 0 :
            ## Min Max Scaling
                timestamp_kd = (timestamp_kd - min(timestamp_kd))/(max(timestamp_kd) - min(timestamp_kd))
            
            if len(timestamp_ku) > 0 :
                timestamp_ku = (timestamp_ku - min(timestamp_ku))/(max(timestamp_ku) - min(timestamp_ku))
            
            ku_count = 0
            kd_count = 0
            
            for i in range(len(data[key]["keyboard_data"])) :
            ## Swap 0 element with 1 element
                temp = list_[i][0]
                list_[i][0] = list_[i][1]
                list_[i][1] = temp
            
            if list_[i][0].lower() == "ku" :
                list_[i][2] = float(timestamp_ku[ku_count])
                ku_count += 1
            else :
                list_[i][2] = float(timestamp_kd[kd_count])
                kd_count += 1

            data[key]["keyboard_data"] = list_[1:]
            data[key] = self.divide_into_batches(data[key]["keyboard_data"])

        return data

    def process_data(self, data,verbose=0) :
        for key in data.keys() :
            if verbose == 1 :
                data[key] = json.loads(data[key])

            timestamp_kd = []
            timestamp_ku = []
            list_ = data[key]["keyboard_data"]

            for i in range(len(data[key]["keyboard_data"])) :
                if list_[i][0] == "KD" :
                    timestamp_kd.append(list_[i][2] - list_[i-1][2] if i > 0 else list_[i][2])
                else :
                    timestamp_ku.append(list_[i][2] - list_[i-1][2] if i > 0 else list_[i][2])
                
            timestamp_kd = np.array(timestamp_kd, dtype=np.float32)
            timestamp_ku = np.array(timestamp_ku, dtype=np.float32)

            if len(timestamp_kd) > 0 :
                ## Min Max Scaling
                timestamp_kd = (timestamp_kd - min(timestamp_kd))/(max(timestamp_kd) - min(timestamp_kd))

            if len(timestamp_ku) > 0 :
                timestamp_ku = (timestamp_ku - min(timestamp_ku))/(max(timestamp_ku) - min(timestamp_ku))

            ku_count = 0
            kd_count = 0

            for i in range(len(data[key]["keyboard_data"])) :
                if list_[i][0] == "KU" :
                    list_[i][2] = float(timestamp_ku[ku_count])
                    ku_count += 1
                else :
                    list_[i][2] = float(timestamp_kd[kd_count])
                    kd_count += 1

            data[key]["keyboard_data"] = list_[1:]
            data[key] = self.divide_into_batches(data[key]["keyboard_data"])

        return data
    ## Replace the character with the ascii value, KD : 0, KU : 1, timestamp : relative
    def convert_list(self, list_) :
        """
        Converts a list of key events into a transformed list.

        Args:
            list_ (list): A list of key events, where each event is represented as a tuple of three elements: 
                        the key action ('KD' for Key Down or 'KU' for Key Up), the key value, and the timestamp.

        Returns:
            list: A transformed list where each event is represented as a list with the following elements:
                - 0 for Key Down or 1 for Key Up
                - The normalized value of the key (if applicable)
                - The timestamp difference between the current event and the previous event

        """
        start_time = 0
        trans_list = []
        timestamp_list_kd = []
        timestamp_list_ku = []
        shift_count = 0

        for i in range(len(list_)) :
            temp = []

            ## Assigning value to Key Up and Key Down
            if list_[i][0] == 'KD' :
                temp.append(0)
            else:
                temp.append(1)

            ## Convert to ascii value
            if (len(str(list_[i][1]).lower()) > 1 and isinstance(list_[i][1],int)) or list_[i][1].lower().find("oem") != -1 or list_[i][1].lower().find("lbutton,") != -1:
                continue
            else :
                temp.append(self.keyboard_map[str(list_[i][1]).lower()]/255)
            
            if str(list_[i][1]).lower() == 'shift' :
                shift_count += 1

            ## Store the diff in timestamp
            if i>=0 :
                temp.append(float(list_[i][2]))
        
            if shift_count < len(list_)*0.2 :    
                trans_list.append(temp)

        return trans_list

    def convert_data(self, data):
        """
        Convert the given data dictionary into a new dictionary with converted lists.

        Args:
            data (dict): The input data dictionary.

        Returns:
            dict: The converted data dictionary.

        """
        p_data = {}

        for key in data.keys():
            list_compiled = []

            for list_ in data[key]:
                temp = self.convert_list(list_)

                if len(temp) > 0:
                    list_compiled.append(temp)

            if len(list_compiled) > 0:
                p_data[key] = list_compiled

        return p_data

    def pad_clip_seq(self, x) :
        # print(x.shape)
        curr_mask = [1]*len(x)

        if(len(x) > self.M) :
        ## If length is greater than the sequence length M : Clip the sequence
            x = x[:self.M]
            curr_mask = curr_mask[:self.M]

        ## If length is less than the sequence length M : Pad the sequence 
        for i in range(max(0,self.M-len(x))) :
            x.append([-1,-1,-1])
            curr_mask.append(0)

        return x, curr_mask

    def return_pad_seq(self, data):
        """
        Pad sequences in the given data dictionary and return the padded sequences along with the corresponding masks.

        Args:
        data (dict): A dictionary containing sequences to be padded.

        Returns:
        tuple: A tuple containing the padded sequences and their corresponding masks.

        """
        mask = {}

        for key in data.keys():
            new_list = []
            mask_l = []

            for list_ in data[key]:
                list_, curr_mask = self.pad_clip_seq(list_)

                new_list.append(list_)
                mask_l.append(curr_mask)

            data[key] = new_list
            mask[key] = mask_l

        return data, mask
    
    ## Combine Pairs of a given set of fixed and free data
    def combine_pairs(self, fixed_data,free_data,mask_fixed,mask_free) :
        data = {}
        mask = {}
        y_data = {}

        for key in fixed_data.keys() :
            if key not in free_data.keys() :
                continue
            else: 
                data[key] = []
                mask[key] = []
                y_data[key] = []

                ## For each user, we create pairs of fixed and free data, Label : 1
                for fixed_index in range(len(fixed_data[key])) :
                    for free_index in range(len(free_data[key])) :
                        data[key].append([fixed_data[key][fixed_index],free_data[key][free_index]])
                        mask[key].append([mask_fixed[key][fixed_index],mask_free[key][free_index]])
                        y_data[key].append(1)

                    for fixed_index_2 in range(len(fixed_data[key])) :
                        if fixed_data[key][fixed_index_2] == fixed_data[key][fixed_index] or ([fixed_data[key][fixed_index_2],fixed_data[key][fixed_index]] in data[key]) or ([fixed_data[key][fixed_index],fixed_data[key][fixed_index_2]] in data[key]):
                            continue
                        else :
                            data[key].append([fixed_data[key][fixed_index],fixed_data[key][fixed_index_2]])
                            mask[key].append([mask_fixed[key][fixed_index],mask_fixed[key][fixed_index_2]])
                            y_data[key].append(0)
                
                ## For each user, we create pairs of fixed and free data, Label : 0
                for free_index in range(len(free_data[key])) :
                    for free_index_2 in range(len(free_data[key])) :
                        if free_data[key][free_index_2] == free_data[key][free_index] or ([free_data[key][free_index_2],free_data[key][free_index]] in data[key]) or ([free_data[key][free_index],free_data[key][free_index_2]] in data[key]):
                            continue
                        else :
                            data[key].append([free_data[key][free_index],free_data[key][free_index_2]])
                            mask[key].append([mask_free[key][free_index],mask_free[key][free_index_2]])
                            y_data[key].append(0)

        return data,y_data,mask

    def combine(self, new_data, mask, y_data):
        """
        Combines the data, mask, and y_data into lists.

        Args:
            new_data (dict): A dictionary containing the new data.
            mask (dict): A dictionary containing the mask.
            y_data (dict): A dictionary containing the y_data.

        Returns:
            tuple: A tuple containing the combined data list, combined mask list, and y_data list.
        """
        combined_data_list = []
        combined_mask_list = []
        y_data_list = []

        for key in new_data.keys():
            if len(combined_data_list) == 0:
                combined_data_list = new_data[key]
                combined_mask_list = mask[key]
                y_data_list = y_data[key]
            else:
                combined_data_list.extend(new_data[key])
                combined_mask_list.extend(mask[key])
                y_data_list.extend(y_data[key])

        return combined_data_list, combined_mask_list, y_data_list
