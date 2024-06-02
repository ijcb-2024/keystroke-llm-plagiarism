# Keystroke Dynamics Against Academic Dishonesty
The following repository contains the Pipelines for conducting evaluations using keystrokes for different scenarios.

The access to all the Pipelines and Datasets has been provided in this repository.

## Evaluation Scenarios

### User-Specific & Agnostic Scenario
For user based scenarios the pipeline User Pipelines Folder which contains the data preprocessing and you can create any combination of datasets for training and testing for different user combinations.

### Keyboard-Specific & Agnostic Scenario
For keyboard based scenarios the folder Keyboard Pipelines contains the preprocessing of Buffalo Dataset keyboard-wise for conducting analysis based on keyboard type.

### Context-Specific & Agnostic Scenario
For context based analysis, the folder Context Pipelines contains the data preprocessing based on context in the SBU Dataset.

### Inter & Intra Dataset Scenarios
For dataset based analysis, the folder Dataset Pipelines contains the data preprocessing for each dataset and can be combined for analysis for any combination.


## Running Pipelines
1. Create a new python venv using ```requirements.txt``` file in the main directory. 

   ```
   conda create --name <env_name> --file requirements.txt
   ```

2. Loading Dataset <br/>
   In order to load the dataset unzip the given datasets. Modify the following global variable for setting the dataset path in the jupyter notebook.
   ```
   ## Root Dataset Directory
   
   ROOT = "../Proposed Dataset"
   ```

3. In each folder the specific and agnostic scenarios are accomodated in each notebook labelled as the same. For trying different combinations of datasets the following code snippet can be modified : 

    **Specific Scenario** <br/>
    The following code snippet ensures that we train and test on SBU dataset with the training set and testing set having disjoint set of sequences from the dataset split at random.
    ````
    '''
        We can create any combination of datasets for training and 
        testing in this pipeline to create the training and testing sets.
        
        fixed_data, free_data --> proposed dataset
        buffalo_fixed, buffalo_free --> buffalo dataset
        gm_fixed, gm_free --> SBU (Gay Marriage)
        gnc_fixed, gnc_free --> SBU (Gun Control)
        rs_fixed, rs_free --> SBU (Restaurant Reviews)
    '''
    
    ## Train : SBU Dataset, Test: Proposed Dataset
    fixed_data_test = {}
    free_data_test = {}
    
    ## For specific case combine the datasets into the training set only. Leave the test set empty.
    
    fixed_data_train = {}
    free_data_train = {}
    
    fixed_data_train.update(gm_fixed)
    fixed_data_train = update_dict(fixed_data_train,rs_fixed)
    fixed_data_train = update_dict(fixed_data_train,gnc_fixed)
    
    free_data_train.update(gm_free)
    free_data_train = update_dict(free_data_train,rs_free)
    free_data_train = update_dict(free_data_train,gnc_free)
    ````
    
    **Agnostic Scenario** <br />
    The following code snippet ensures that we train on the topics with Gun Control & Restaurant Feedback as context and train on sequences with Gay Marriage as the context with the training set and testing set having disjoint set of sequences from the dataset split at random.
    ````
    '''
        We can create any combination of datasets for training and 
        testing in this pipeline to create the training and testing sets.
        
        fixed_data, free_data --> proposed dataset
        buffalo_fixed, buffalo_free --> buffalo dataset
        gm_fixed, gm_free --> SBU (Gay Marriage)
        gnc_fixed, gnc_free --> SBU (Gun Control)
        rs_fixed, rs_free --> SBU (Restaurant Reviews)
    '''
    
    ## Train : SBU Dataset, Test: Proposed Dataset
    fixed_data_test = {}
    free_data_test = {}
    
    fixed_data_test.update_dict(gm_fixed)
    free_data_test.update_dict(gm_free)
    
    ## For specific case combine the datasets into the training set only. Leave the test set empty.
    
    fixed_data_train = {}
    free_data_train = {}
    
    fixed_data_train = update_dict(fixed_data_train,rs_fixed)
    fixed_data_train = update_dict(fixed_data_train,gnc_fixed)
    
    free_data_train = update_dict(free_data_train,rs_free)
    free_data_train = update_dict(free_data_train,gnc_free)
    ````
4. Global Variables <br/>
   In order to run custom experiments and fine tune the models modify these global variables in the notebook for running different versions of the model and optimising the performance.
   ```
   M -- Sequence Length for each input sequence in the model
   BATCH_SIZE -- Batch Size for model input
   LR -- Learning Rate
   EPOCHS -- Number of epochs to train the model
   DROPOUT -- Model Dropout
   ```

## Results

### Specific Scenarios
| **Train** | **Test** | **Acc.** | **F<sub>1</sub>** | **FAR** | **FRR** |
|-----------|----------|----------|------------------|---------|---------|
| **Keyboard Specific** | | | | | |
| $K_0$ | $K_0$ | 84.64 | 83.45 | 25.38 | 19.83 |
| $K_1$ | $K_1$ | 80.02 | 78.91 | 29.11 | 24.19 |
| $K_2$ | $K_2$ | 77.77 | 76.58 | 32.14 | 25.12 |
| $K_3$ | $K_3$ | 74.98 | 73.34 | 32.60 | 30.16 |
| **Context Specific** | | | | | |
| GM | GM | 80.24 | 81.52 | 30.81 | 21.27 |
| GC | GC | 79.39 | 80.67 | 34.62 | 22.33 |
| RF | RF | 76.52 | 78.01 | 34.02 | 31.13 |
| **User Specific** | | | | | |
| Merged | Merged | 81.86 | 81.85 | 23.71 | 26.24 |
| **Dataset Specific** | | | | | |
| S | S | 80.85 | 83.31 | 18.83 | 32.93 |
| P | P | 81.04 | 82.16 | 22.60 | 28.20 |
| B | B | 85.72 | 84.72 | 17.64 | 23.91 |



### Agnostic Scenarios
| **Train** | **Test** | **Acc.** | **F<sub>1</sub>** | **FAR** | **FRR** |
|-----------|----------|----------|------------------|---------|---------|
| **Keyboard Agnostic** | | | | | |
| $K_{0,1,2}$ | $K_3$ | 78.11 | 75.88 | 28.04 | 29.01 |
| $K_{0,1,3}$ | $K_2$ | 79.71 | 78.06 | 27.50 | 28.37 |
| $K_{0,2,3}$ | $K_1$ | 80.54 | 78.77 | 24.01 | 28.37 |
| $K_{1,2,3}$ | $K_0$ | 78.96 | 76.89 | 29.15 | 24.95 |
| **Context Agnostic** | | | | | |
| (GM, RF) | GC | 72.96 | 72.40 | 38.30 | 28.56 |
| (GC, RF) | GM | 78.67 | 78.24 | 32.30 | 24.10 |
| (GM, GC) | RF | 70.21 | 70.23 | 39.65 | 34.73 |
| **User Agnostic** | | | | | |
| Merged (80-10-10) | | 63.56 | 62.98 | 42.51 | 39.57 |
| Merged (50-25-25) | | 66.54 | 66.22 | 38.82 | 39.86 |
| **Dataset Agnostic** | | | | | |
| (S, P) | B | 68.72 | 66.21 | 33.13 | 39.66 |
| (S, B) | P | 73.23 | 72.65 | 27.95 | 40.22 |
| (P, B) | S | 52.24 | 61.86 | 47.53 | 48.51 |
| S | (P, B) | 59.73 | 57.03 | 41.36 | 44.29 |
| P | (S, B) | 56.17 | 54.13 | 44.95 | 45.72 |
| B | (S, P) | 53.57 | 53.16 | 46.32 | 48.01 |


The following are the results for all evaluation scenarios which can be replicated by running the following pipelines. 
