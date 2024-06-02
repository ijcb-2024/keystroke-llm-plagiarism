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


