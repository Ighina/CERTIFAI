# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:37:55 2020

@author: Iacopo
"""
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import manhattan_distances as L1
from sklearn.metrics.pairwise import euclidean_distances as L2
from skimage.metrics import structural_similarity as SSIM
from tqdm import tqdm


class CERTIFAI:
    def __init__(self, Pm = .2, Pc = .5, dataset_path = None,
                 numpy_dataset = None):
        """The class instance is initialised with the probabilities needed
        for the counterfactual generation process and an optional path leading
        to a .csv file containing the training set. If the path is provided,
        the class will assume in some of its method that the training set is tabular
        in nature and pandas built-in functions will be used in several places, instead
        of the numpy or self defined alternatives."""
        
        self.Pm = Pm
        
        self.Pc = Pc
        
        self.Population = None
        
        self.distance = None
        
        if dataset_path is not None:
            self.tab_dataset = pd.read_csv(dataset_path)
            
        else:
            self.tab_dataset = None
            
            if numpy_dataset is not None:
                self.tab_dataset = numpy_dataset
            
        self.constraints = None
        
        self.predictions = None
        
        self.results = None
    
    @classmethod
    def from_csv(cls, path):
        return cls(dataset_path=path)
    
    def change_Pm(self, new_Pm):
        '''Set the new probability for the second stage of counterfactual
        generation.
        Arguments:
            Inputs:
                new_Pm: new probability of picking a sample from the counterfactuals
                to be changed as described in the original paper for the second
                step of the generation process.
                
            Outputs:
                None, the Pm attribute is changed'''
        
        self.Pm = new_Pm
        
    def change_Pc(self, new_Pc):
        '''Set the new probability for the third stage of counterfactual
        generation.
        Arguments:
            Inputs:
                new_Pc: new probability of picking a sample from the counterfactuals
                to be changed as described in the original paper for the third
                step of the generation process.
                
            Outputs:
                None, the Pc attribute is changed'''
        
        self.Pc = new_Pc
        
    def get_con_cat_columns(self, x):
        
        assert isinstance(x, pd.DataFrame), 'This method can be used only if input\
            is an instance of pandas dataframe at the moment.'
        
        con = []
        cat = []
        
        for column in x:
            if x[column].dtype == 'O':
                cat.append(column)
            else:
                con.append(column)
                
        return con, cat
        
    def Tab_distance(self, x, y, continuous_distance = 'L1', con = None,
                     cat = None):
        """Distance function for tabular data, as described in the original
        paper. This function is the default one for tabular data in the paper and
        in the set_distance function below as well. For this function to be used,
        the training set must consist of a .csv file as specified in the class
        instatiation above. This way, pandas can be used to infer whether a
        feature is categorical or not based on its pandas datatype and, as such, it is important that all columns
        in the dataframe have the correct datatype.
        
        Arguments:
            x (pandas.dataframe): the input sample from the training set. This needs to be
            a row of a pandas dataframe at the moment, but the functionality of this
            function will be extended to accept also numpy.ndarray.
            
            y (pandas.dataframe or numpy.ndarray): the comparison samples (i.e. here, the counterfactual)
            which distance from x needs to be calculated.
            
            continuous_distance (bool): the distance function to be applied
            to the continuous features. Default is L1 function.
            
            con (list): list of the continuous features (i.e. columns) names
            
            cat (list): list of the categorical features (i.e. columns) names
        """
        
        assert isinstance(x, pd.DataFrame), 'This distance can be used only if input\
            is a row of a pandas dataframe at the moment.'
            
        
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns = x.columns.tolist())
        else:
            y.columns = x.columns.tolist()
        
        if con is None or cat is None:
            con, cat = self.get_con_cat_columns(x)
        
        if len(cat)>0:
            
            cat_distance = len(cat) - (x[cat].values == y[cat].values).sum(axis = 1)
        
        else:
            
            cat_distance = 1
            
        if len(con)>0:
            
            if continuous_distance == 'L1':
                con_distance = L1(x[con], y[con])
                
            else:
                con_distance = L2(x[con], y[con])
                
        else:
            con_distance = 1
            
        return len(con)/x.shape[-1]*con_distance + len(cat)/x.shape[-1]*cat_distance
    
    def img_distance(self, x, y):
        
        distances = []
        
        for counterfact in y:
            distances.append(SSIM(x, counterfact))
        
        return np.array(distances).reshape(1,-1)
            
        
        
    def set_distance(self, kind = 'automatic', x = None):
        """Set the distance function to be used in counterfactual generation.
        The distance function can either be manually chosen by passing the 
        relative value to the kind argument or it can be inferred by passing the
        'automatic' value.
        
        Arguments:
            Inputs:
                kind (string): possible values, representing the different
                distance functions are: 'automatic', 'L1', 'SSIM', 'L2' and 'euclidean' (same as L2)
                
                x (numpy.ndarray or pandas.DataFrame): training set or a sample from it on the basis
                of which the function will decide what distance function to use if kind=='automatic'.
                Specifically, if the input is tridimensional, the function will choose the distance
                function more suitable for images (SSIM), if the training set is a .csv file then
                the function more suitable for tabular data will be used, otherwise the function
                will backoff to using L1 norm distance.
                
            Outputs:
                None, set the distance attribute as described above."""
        
        
        if kind == 'automatic':
            
            assert x is not None or self.tab_dataset is not None, 'For using automatic distance assignment,\
                the input data needs to be provided or the class needs to be initialised with a csv file!'
            
            if x is None:
                x = self.tab_dataset
            
            if len(x.shape)>2:
                self.distance = self.img_distance
                
                print('SSIM distance has been set as default')
                
            else:
                con, cat = self.get_con_cat_columns(x)
                if len(cat)>0:
                    self.distance = self.Tab_distance
                    print('Tabular distance has been set as default')
                    
                else:
                    self.distance = L1
                    print('L1 norm distance has been set as default')
                
        elif kind == 'tab_distance':
            self.distance = self.Tab_distance
        elif kind == 'L1':
            self.distance = L1
        elif kind == 'SSIM':
            self.distance = self.img_distance
        elif kind == 'L2':
            self.distance = L2
        elif kind == 'euclidean':
            self.distance = L2
        else:
            raise ValueError('Distance function specified not recognised:\
                             use one of automatic, L1, SSIM, L2 or euclidean.')
        
    def set_population(self, x=None):
        """Set the population limit (i.e. number of counterfactuals created at each generation).
        following the original paper, we define the maximum population as the minum between the squared number of features
        to be generated and 30000.
        
        Arguments:
            Inputs:
                x (numpy.ndarray or pandas.DataFrame): the training set or a sample from it, so that the number of features can be obtained.
            
            Outputs:
                None, the Population attribute is set as described above
        """
        
        if x is None:
            assert self.tab_dataset is not None, 'If input is not provided, the class needs to be instatiated\
                with an associated csv file, otherwise there is no input data for inferring population size.'
            
            x = self.tab_dataset
        
        if len(x.shape)>2:
            self.Population = min(sum(x.shape[1:])**2, 30000)
        else:
            self.Population = min(x.shape[-1]**2, 30000)
        
    def set_constraints(self, x = None, fixed = None):
        '''Set the list of constraints for each input feature, whereas
        each constraint consist in the minimum and maximum value for 
        the given continuous feature. If a categorical feature is encountered,
        then the number of unique categories is appended to the list instead.
        
        Arguments:
            Inputs:
            x (numpy.ndarray): if the training set is not a pandas dataframe (see above),
            this function will expect a numpy array with the entire training set in the
            form of a numpy array.
            
            fixed (list): a list of features to be kept fixed in counterfactual generation 
            (i.e. all counterfactual will have the same value as the given sample for that
             feature). If no list is provided, then no feature will be kept fixed 
            
            Outputs:
                None, an attribute 'constraints' is created for the class, where
                the constraints are stored.
            '''
        
        fixed_feats = set() if fixed is None else set(fixed)
        
        self.constraints = []
        
        if x is None:
            x = self.tab_dataset
        
        if len(x.shape)>2:
            x = self.tab_dataset if self.tab_dataset is not None else x.copy()
            
            x = pd.DataFrame(x.reshape(x.shape[0], -1))
            
        if isinstance(x, pd.DataFrame):
            for i in x:
                if i in fixed_feats:
                    # Placeholder if the feature needs to be kept fixed in generating counterfactuals
                    self.constraints.append(i)
                # Via a dataframe is also possible to constran categorical fatures (not supported for numpy array)
                elif x.loc[:,i].dtype == 'O':
                    self.constraints.append((0, len(pd.unique(x.loc[:,i]))))
                else:
                    self.constraints.append((min(x.loc[:,i]), max(x.loc[:,i])))
        
        else:
            assert x is not None, 'A numpy array should be provided to get min-max values of each column,\
                or, alternatively, a .csv file needs to be supplied when instatiating the CERTIFAI class'
            
            for i in range(x.shape[1]):
                if i in fixed_feats:
                    # Placeholder if the feature needs to be kept fixed in generating counterfactuals
                    self.constraints.append(i)
                else:
                    self.constraints.append((min(x[:,i]), max(x[:, i])))
                
        
    
    def transform_x_2_input(self, x, pytorch = True):
        '''Function to transform the raw input in the form of a pandas dataset
        or of a numpy array to the required format as input of the neural net(s)
        
        Arguments:
            Inputs:
                x (pandas.DataFrame or numpy.ndarray): the "raw" input to be
                transformed.
                
                torch (bool): the deep learning library
                used for training the model analysed. Options are torch==True for 
                pytorch and torch==False for tensorflow/keras
                
            Outputs:
                transformed_x (torch.tensor or numpy.ndarray): the transformed
                input, ready to be passed into the model.'''
                
        if isinstance(x, pd.DataFrame):
            
            x = x.copy()
            
            con, cat = self.get_con_cat_columns(x)
            
            if len(cat)>0:
                
                for feature in cat:
                    
                    enc = LabelEncoder()
                    
                    x[feature] = enc.fit(x[feature]).transform(x[feature])
            
            model_input = torch.tensor(x.values, dtype=torch.float) if pytorch else x.values
            
        elif isinstance(x, np.ndarray):
            
            model_input = torch.tensor(x, dtype = torch.float) if pytorch else x
            
        else:
            raise ValueError("The input x must be a pandas dataframe or a numpy array")
            
        return model_input
    
    def generate_prediction(self, model, model_input, pytorch=True, classification = True):
        '''Function to output prediction from a deep learning model
        
        Arguments:
            Inputs:
                model (torch.nn.Module or tf.Keras.Model): the trained deep learning
                model.
                
                x (torch.tensor or numpy.ndarray): the input to the model.
                
                pytorch (bool): whether pytorch or keras is used.
                
                classification (bool): whether a classification or regression task is performed.
                
            Output
                prediction (numpy.ndarray): the array containing the single greedily predicted
                class (in the case of classification) or the single or multiple predicted value
                (when classification = False).
                '''
        
        if classification:
            if pytorch:
                with torch.no_grad():
                    prediction = np.argmax(model(model_input).numpy(), axis = -1)
            else:
                prediction = np.argmax(model.predict(model_input), axis = -1)
                
        else:
            if pytorch:
                with torch.no_grad():
                    prediction = model(model_input).numpy()
            else:
                prediction = model.predict(model_input)
                
        return prediction
    
    def generate_counterfacts_list_dictionary(self, counterfacts_list,
                                              distances, fitness_dict,
                                              retain_k, start=0):
        '''Function to generate and trim at the same time the list containing
        the counterfactuals and a dictionary having fitness score
        for each counterfactual index in the list. 
        
        Arguments:
            Inputs:
                counterfacts_list (list): list containing each counterfactual
                
                distances (numpy.ndarray): array containing distance from sample
                for each counterfactual
                
                fitness_dict (dict): dictionary containing the fitness score
                (i.e. distance) for each counterfactual index in the list
                as key. If an empty dictionary is passed to the function, then
                the index of the counterfactual starts from 0, else it starts
                counting from the value of the start argument.
                
                start (int): index from which to start assigning keys for
                the dictionary
                
            Outputs:
                selected_counterfacts (list): list of top counterfacts from
                the input list, selected on the basis of their fitness score
                and having length=retain_k
                
                fitness_dict (dict): dictionary of fitness scores stored
                by the relative counterfactual index.'''
                
        gen_dict = {i:distance for i, 
                                distance in enumerate(distances)}
                    
        gen_dict = {k:v for k,v in sorted(gen_dict.items(),
                                          key = lambda item: item[1])}
        
        selected_counterfacts = []
        
        k = 0
        
        for key,value in gen_dict.items():
            
            if k==retain_k:
                break
            selected_counterfacts.append(counterfacts_list[key])
            
            fitness_dict[start+k] = value
            
            k+=1
            
        return selected_counterfacts, fitness_dict
    
    def generate_cats_ids(self, dataset = None, cat = None):
        '''Generate the unique categorical values of the relative features
        in the dataset.
        
        Arguments:
            Inputs:
            
            dataset (pandas.dataframe): the reference dataset from which to extract
            the categorical values. If not provided, the function will assume that
            a dataframe has been saved in the class instance when created, via the
            option for initialising it from a csv file.
            
            cat (list): list of categorical features in the reference dataset. If
            not provided, the list will be obtained via the relative function of
            this class.
            
            Output:
                
            cat_ids (list): a list of tuples containing the unique categorical values
            for each categorical feature in the dataset, their number and the relative
            column index.
        '''
        if dataset is None:
            assert self.tab_dataset is not None, 'If the dataset is not provided\
            to the function, a csv needs to have been provided when instatiating the class'
            
            dataset = self.tab_dataset
            
        if cat is None:
            con, cat = self.get_con_cat_columns(dataset)
            
        cat_ids = []
        for index, key in enumerate(dataset):
            if key in set(cat):
                cat_ids.append((index,
                                len(pd.unique(dataset[key])),
                                pd.unique(dataset[key])))
        return cat_ids
    
    def generate_candidates_tab(self, 
                                sample,
                                normalisation = None, 
                                constrained = True,
                                has_cat = False,
                                cat_ids = None,
                                img = False):
        '''Function to generate the random (constrained or unconstrained)
        candidates for counterfactual generation if the input is a pandas
        dataframe (i.e. tabular data).
        
        Arguments:
            Inputs:
                sample (pandas.dataframe): the input sample for which counterfactuals
                need to be generated.
                
                normalisation (str) ["standard", "max_scaler"]: the
                normalisation technique used for the data at hand. Default
                is None, while "standard" and "max_scaler" techniques are
                the other possible options. According to the chosen value
                different random number generators are used.
                
                constrained (bool): whether the generation of each feature
                will be constrained by its minimum and maximum value in the
                training set (it applies just for the not normalised scenario
                              as there is no need otherwise)
                
                has_cat (bool): whether the input sample includes categorical
                variables or not (they are treated differently in the distance
                                  function used in the original paper).
                
                cat_ids (list): list of the names of columns containing categorical
                values.
                
            Outputs:
                generation (list): a list of the random candidates generated.
                
                distances (numpy.ndarray): an array of distances of the candidates
                from the current input sample.
                '''
        
        nfeats = sample.shape[-1]
        
        if normalisation is None:
                            
            if constrained:
                
                generation = []
                
                temp = []
                for constraint in self.constraints:
                    if not isinstance(constraint, tuple):
                        # In this case the given feature is fixed
                        temp.append(sample.loc[:,constraint].values)
                    else:
                        temp.append(np.random.randint(constraint[0]*100, (constraint[1]+1)*100,
                                                      size = (self.Population, 1))/100
                                    )
                
                generation = np.concatenate(temp, axis = -1)
                
            else:
                # If not constrained, we still don't want to generate values that are not totally unrealistic
                
                low = min(sample)
                
                high = max(sample)
                
                generation = np.random.randint(low,high+1, size = (self.Population, nfeats))
            
            
        elif normalisation == 'standard':
            generation = np.random.randn(self.Population, nfeats)
                
        elif normalisation == 'max_scaler':
            generation = np.random.rand(self.Population, nfeats)
        
        else:
            raise ValueError('Normalisation option not recognised:\
                             choose one of "None", "standard" or\
                                 "max_scaler".')
        
        if has_cat:
            
            assert cat_ids is not None, 'If categorical features are included in the dataset,\
                the relative cat_ids (to be generated with the generate_cats_ids method) needs\
                    to be provided to the function.'
            
            generation = pd.DataFrame(generation, columns=sample.columns.tolist())
            
            for idx, ncat, cat_value in cat_ids:
                
                random_indeces = np.random.randint(0, ncat, size = self.Population)
                
                random_cats = [cat_value[feat] for feat in random_indeces]
                
                generation.iloc[:, idx] = random_cats
                
            distances = self.distance(sample, generation)[0]
            
            generation = generation
        
        else:
            distances = self.distance(sample, generation)[0]
            
            generation = generation.tolist()
        
            generation = pd.DataFrame(generation)
        
        for i in sample:
            generation[i] = generation[i].astype(sample[i].dtype)
        
        
        return generation.values.tolist(), distances
                    
    
    def mutate(self, counterfacts_list):
        '''Function to perform the mutation step from the original paper
        
        Arguments:
            Input:
                counterfacts_list (list): the candidate counterfactuals
                from the selection step.
                
            Output:
                mutated_counterfacts (numpy.ndarray): the mutated candidate
                counterfactuals.'''
        
        nfeats = len(counterfacts_list[0])
        
        dtypes = [type(feat) for feat in counterfacts_list[0]]
        
        counterfacts_df = pd.DataFrame(counterfacts_list)
        
        random_indeces = np.random.binomial(1, self.Pm, len(counterfacts_list))
        
        mutation_indeces = [index for index, i in enumerate(random_indeces) if i]
        
        for index in mutation_indeces:
            mutation_features = np.random.randint(0, nfeats, 
                                                  size = np.random.randint(1, nfeats))
            
            for feat_ind in mutation_features:
                if isinstance(counterfacts_df.iloc[0, feat_ind], str):
                    counterfacts_df.iloc[index, feat_ind] = np.random.choice(
                        np.unique(counterfacts_df.iloc[:, feat_ind]))
                    
                else:
                    counterfacts_df.iloc[index, feat_ind] = 0.5*(
                        np.random.choice(counterfacts_df.iloc[:, feat_ind]) +
                    np.random.choice(counterfacts_df.iloc[:, feat_ind]))
        
        for index, key in enumerate(counterfacts_df):
            counterfacts_df[key] = counterfacts_df[key].astype(dtypes[index])
        
        return counterfacts_df.values.tolist()
    
    def crossover(self, counterfacts_list, return_df = False):
        '''Function to perform the crossover step from the original paper
        
        Arguments:
            Input:
                counterfacts_list (list): the candidate counterfactuals
                from the mutation step.
                
            Output:
                crossed_counterfacts (numpy.ndarray): the changed candidate
                counterfactuals.'''
        
        nfeats = len(counterfacts_list[0])
        
        random_indeces = np.random.binomial(1, self.Pc, len(counterfacts_list))
        
        mutation_indeces = [index for index, i in enumerate(random_indeces) if i]
        
        counterfacts_df = pd.DataFrame(counterfacts_list)
        
        while mutation_indeces:
            
            individual1 = mutation_indeces.pop(np.random.randint(0, len(mutation_indeces)))
            
            if len(mutation_indeces)>0:
                
                individual2 = mutation_indeces.pop(np.random.randint(0, len(mutation_indeces)))
                
                mutation_features = np.random.randint(0, nfeats, 
                                                      size = np.random.randint(1, nfeats))
                
                features1 = counterfacts_df.iloc[individual1, mutation_features]
                
                features2 = counterfacts_df.iloc[individual2, mutation_features]
                
                counterfacts_df.iloc[individual1, mutation_features] = features2
                
                counterfacts_df.iloc[individual2, mutation_features] = features1
        
        if return_df:
            return counterfacts_df
        
        return counterfacts_df.values.tolist()
    
    def fit(self, 
            model,
            x = None,
            model_input = None,
            pytorch = True,
            classification = True,
            generations = 3, 
            distance = 'automatic',
            constrained = True,
            class_specific = None,
            select_retain = 1000,
            gen_retain = 500,
            final_k = 1,
            normalisation = None,
            fixed = None,
            verbose = False):
        
        if x is None:
            assert self.tab_dataset is not None, 'Either an input is passed into\
            the function or a the class needs to be instantiated with the path\
                to the csv file containing the dataset'
            
            x = self.tab_dataset
            
        else:
            
            x = x.copy() 
            
        if self.constraints is None:
                self.set_constraints(x, fixed)
            
        if self.Population is None:
                self.set_population(x)
                
        if self.distance is None:
                self.set_distance(distance, x)
                
        if model_input is None:
            model_input = self.transform_x_2_input(x, pytorch = pytorch)
            
        if torch:
            model.eval()
        
        if self.predictions is None: 
            self.predictions = self.generate_prediction(model, model_input,
                                                  pytorch = pytorch,
                                                  classification = classification)
        
        if len(x.shape)>2:
            
            x = x.reshape(x.shape[0], -1)
        
        self.results = []
        
        if isinstance(x, pd.DataFrame):
                
            con, cat = self.get_con_cat_columns(x)
            
            has_cat = True if len(cat)>0 else False
            
            cat_ids = None
            
            if has_cat:
                cat_ids = self.generate_cats_ids(x)
        
        else:
             x = pd.DataFrame(x)   
        
        
        
        if classification and class_specific is not None:
            x = x.iloc[self.predictions == class_specific]
            self.class_specific = class_specific
                
                
               
        tot_samples = tqdm(range(x.shape[0])) if verbose else range(x.shape[0])
        
        
        for i in tot_samples:
            
            if verbose:
                tot_samples.set_description('Generating counterfactual(s) for sample %s' % i)
            
            sample = x.iloc[i:i+1,:]
            
            counterfacts = []
            
            counterfacts_fit = {}
            
            for g in range(generations):
            
                generation, distances = self.generate_candidates_tab(sample,
                                                                normalisation,
                                                                constrained,
                                                                has_cat,
                                                                cat_ids)
                    
                selected_generation, _ = self.generate_counterfacts_list_dictionary(
                    counterfacts_list = generation,
                    distances = distances, 
                    fitness_dict = {},
                    retain_k = select_retain, 
                    start=0)
                
                selected_generation = np.array(selected_generation)
                
                mutated_generation = self.mutate(selected_generation)
                
                crossed_generation = self.crossover(mutated_generation, 
                                                    return_df = True)
                
                gen_input = self.transform_x_2_input(crossed_generation,
                                                     pytorch = pytorch)
                
                counter_preds = self.generate_prediction(model,
                                                         gen_input,
                                                         pytorch,
                                                         classification)
                
                diff_prediction = [counter_pred!=self.predictions[g] for
                                   counter_pred in counter_preds]
                
                final_generation = crossed_generation.loc[diff_prediction]
                
                final_distances = self.distance(sample, final_generation)[0]
                
                final_generation, counterfacts_fit = self.generate_counterfacts_list_dictionary(
                    counterfacts_list = final_generation.values.tolist(),
                    distances = final_distances, 
                    fitness_dict = counterfacts_fit,
                    retain_k = gen_retain, 
                    start = len(counterfacts_fit))
                
                counterfacts.extend(final_generation)
                
                assert len(counterfacts)==len(counterfacts_fit), 'Something went wrong: len(counterfacts): {}, len(counterfacts_fit): {}'.format(len(counterfacts), len(counterfacts_fit))
            
            counterfacts, fitness_dict = self.generate_counterfacts_list_dictionary(
                    counterfacts_list = counterfacts,
                    distances = list(counterfacts_fit.values()), 
                    fitness_dict = {},
                    retain_k = final_k, 
                    start = 0)
            
            self.results.append((sample, counterfacts, list(fitness_dict.values())))
            
    def check_robustness(self, x = None, normalised = False):
        '''Calculate the Counterfactual-based Explanation for Robustness
        (CER) score or the normalised CER (NCER) if normalised argument is
        set to true.
        
        Arguments:
            Inputs:
                x (pandas.dataframe or numpy.ndarray): the dataset for which
                the counterfactual were generated
                
                normalised (bool): whether to calculate the normalised score
                or not.
                
            Outputs:
                CERScore (float): a float describing models' robustness (the
                higher the score, the more robust the model)'''
                
        
        assert self.results is not None, 'To run this method counterfactulals need\
        to be generated with the generate_counterfactuals method first.'
        
        distances = np.array([result[2] for result in self.results])
        
        CERScore = distances.mean()
        
        if normalised:
            assert self.predictions is not None, 'To calculate NCER score, predictions\
                for the dataset are needed: by running generate_counterfactuals method\
                    they should be automatically generated!'
            
            if x is None:
                assert self.tab_dataset is not None, 'If normalising, the original\
                    dataset for which the counterfactuals were generated needs to be provided'
                
                x = self.tab_dataset
                
            
            elif isinstance(x, np.ndarray):
                '''If the input is a numpy array (e.g. for an image), we will\
                    transform it into a pandas dataframe for consistency'''
                x = pd.DataFrame(x) 
            
            if len(self.results) < len(x):
                x = x.iloc[self.predictions==self.class_specific]
                predictions = self.predictions[self.predictions==self.class_specific]
            else:
                predictions = self.predictions
            
            unique_preds, preds_prob = np.unique(predictions, return_counts = True)
            
            preds_prob = preds_prob / preds_prob.sum()
            
            normalisation_term = 0
            
            for index, i in enumerate(unique_preds):
                
                current_samples = x.iloc[predictions==i,:]
                if len(current_samples)>1:
                    current_expect_distance = np.array(
                        [self.distance(
                            current_samples.iloc[i:i+1, :],
                        current_samples)[0] for i in range(
                        len(current_samples))]).sum()/(
                            len(current_samples)**2 -
                            len(current_samples))
                        
                    normalisation_term += current_expect_distance * preds_prob[index]
                
            return CERScore/normalisation_term
        
        return CERScore
    
    
    def check_fairness(self, control_groups, x = None, 
                       normalised = False, print_results = False,
                       visualise_results = False):
        '''Calculate the fairness of the model for specific subsets of the dataset
        by comparing the relative CER scores of the different subsets.
        
        Arguments:
            Inputs:
                control_groups (list of dictionaries): a list containing dictionaries
                specifying the condition over which to divide the dataset in subsets.
                Specifically, the key(s) of each dictionary specify a feature of the
                dataset and the related value is the value that such feature must
                have in the subset.
                
                x (pandas.dataframe or numpy.ndarray): the dataset for which
                the counterfactual were generated
                
                normalised (bool): whether to calculate the normalised score
                or not.
                
                print_results (bool): whether to explicitly print the results (
                    including which subset is the most robust and which the least)
                
                visualise_results (bool): whether to plot the results with a 
                standard bar chart from matplotlib.
                
            Outputs:
                results (dictionary): a dictionary containing each splitting
                condition(s) as keys and the relative Robustness scores as values.'''
                
        
        assert self.results is not None, 'To run this method counterfactulals need\
        to be generated with the generate_counterfactuals method first.'
        
        if x is None:
            assert self.tab_dataset is not None, 'If normalising, the original\
                dataset for which the counterfactuals were generated needs to be provided'
            
            x = self.tab_dataset
        
        elif isinstance(x, np.ndarray):
            '''If the input is a numpy array (e.g. for an image), we will\
                transform it into a pandas dataframe for consistency'''
            x = pd.DataFrame(x) 
        
        if len(self.results) < len(x):
            x = x.iloc[self.predictions==self.class_specific]
            x.reset_index(inplace=True)
        
        results = {}
        
        for group in control_groups:
            
            x_new = x.copy()
            
            for feature, condition in group.items():
                x_new = x_new[x_new[feature]==condition]
            
            distances = np.array([result[2] for result in 
                                  self.results])[x_new.index]
        
            CERScore = distances.mean()
            
            if normalised:
                assert self.predictions is not None, 'To calculate NCER score, predictions\
                    for the dataset are needed: by running generate_counterfactuals method\
                        they should be automatically generated!'
                
                
                
                unique_preds, preds_prob = np.unique(self.predictions, return_counts = True)
                
                preds_prob = preds_prob / preds_prob.sum()
                
                normalisation_term = 0
                
                subset_predictions = self.predictions[x_new.index]
                
                for index, i in enumerate(unique_preds):
                    
                    current_samples = x_new.iloc[subset_predictions==i,:]
                    
                    if len(current_samples)>1:
                        current_expect_distance = np.array(
                            [self.distance(
                                current_samples.iloc[i:i+1, :],
                            current_samples)[0] for i in range(
                            len(current_samples))]).sum()/(
                                len(current_samples)**2 -
                                len(current_samples))
                            
                        normalisation_term += current_expect_distance * preds_prob[index]
                
                CERScore /= normalisation_term
            
            key = '-&-'.join([str(k)+'='+str(v) for k,v in group.items()])
            
            results[key] = CERScore
        
        CERScores = np.array(list(results.values()))
        Subsets = np.array(list(results.keys()))
        
        if print_results:
            
            for k, v in results.items():
                msg = re.sub('-&-', ' and ', k)
                
                
                print('CERScore for subset of dataset having {} is {}'.format(
                    msg, v))
            
            
            max_group = Subsets[np.argmax(CERScores)]
            min_group = Subsets[np.argmin(CERScores)]
            
            print('The more robust subset is the one having {}'.format(
                    re.sub('-&-', ' and ', max_group)))
            
            print('The less robust subset is the one having {}\n\
                  with a difference of {} from the more robust:\n\
                      the model might underperform for this group'.format(
                    re.sub('-&-', ' and ', min_group), np.max(CERScores - np.min(CERScores))))
        
        if visualise_results:
            
            plt.bar(Subsets, CERScores, color = 'red')
            
            plt.title('Robustness Score for subsets of the Dataset')
            
            plt.xlabel('Subset')
            
            plt.ylabel('Robustness Score')
            
            if len(Subsets[np.argmax([len(sub) for sub in Subsets])]) + len(Subsets)>25:
                plt.xticks(rotation=90)
            
            plt.show()
        
        return results
    
    
    def check_feature_importance(self, x = None, sensibility = 10,
                                 print_results = False,
                                 visualise_results = False):
        '''Check feature importance by counting how many times each feature has
        been changed to create a counterfactual. For continuous features, the unit
        denoting a change in the feature value is set to be 1 standard deviation
        (in the future more options will be added)
        Arguments:
            Inputs:
                x (pandas.dataframe or numpy.ndarray): the dataset for which
                the counterfactual were generated
                
                sensibility (int or dictionary): scaling value for the standard deviations of
                continuous features. The bigger the sensibility the smaller the
                interval a counterfactual should lay in to be considered equal to
                the real sample. Multiple scaling values can be passed so that
                each continuous feature can have its specific scaling value. In
                this case, the sensibilities need to be passed in a dictionary
                having as key the feature/column name of the interested feature
                and as value the scaling value (i.e. sensibility).
                
                print_results (bool): whether to explicitly print the results
                
                visualise_results (bool): whether to plot the results with a 
                standard bar chart from matplotlib.
        
        '''
        
        assert self.results is not None, 'To run this method counterfactulals need\
        to be generated with the generate_counterfactuals method first.'
        
        if isinstance(self.results[0][0], pd.DataFrame):
            
            if x is None:
                assert self.tab_dataset is not None, 'The original\
                    dataset for which the counterfactuals were generated needs to be provided'
                
                x = self.tab_dataset 
                
                cols = x.columns.tolist()
        
        elif isinstance(x, np.ndarray):
            '''If the input is a numpy array (e.g. for an image), we will\
                transform it into a pandas dataframe for consistency'''
            x = pd.DataFrame(x)    
            
            cols = x.columns.tolist()
        
        if len(self.results) < len(x):
            x = x.iloc[self.predictions==self.class_specific]
        
        if isinstance(sensibility, int):
            sensibility = {k:sensibility for k in cols}
        
        nfeats = x.shape[1]
        
        importances = [0 for i in range(nfeats)]
        
        std = x.describe(include='all').loc['std'].values
        
        counterfactuals = np.array([result[1] for result in self.results])
        
        ncounterfacts = counterfactuals.shape[1]
        
        counterfactuals = counterfactuals.reshape(-1, nfeats)
        
        if x.shape[0]==counterfactuals.shape[0]:
            for index, feature in enumerate(x):
                if np.isnan(std[index]):
                    importances[index] = (x[feature].values !=
                                          counterfactuals[:,index]).sum()
                else:
                    importances[index] = len(x)-((x[feature].values - std[index]/sensibility[feature]<
                                          counterfactuals[:,index].astype(
                                          x[feature].dtype)) & (
                                          counterfactuals[:,index].astype(
                                          x[feature].dtype)<
                                          x[feature].values + std[index]/sensibility[feature])).sum()
        
        else:
            new_x = np.repeat(x.iloc[0,:].values.reshape(1, -1), 
                              ncounterfacts, axis = 0)
            
            for i in range(1, len(x)):
                new_x = np.concatenate((new_x, np.repeat(
                    x.iloc[i,:].values.reshape(1, -1), 
                    ncounterfacts, axis = 0)))
                
            for index, feature in enumerate(x):
                if np.isnan(std[index]):
                    importances[index] = (new_x[:,index] !=
                                          counterfactuals[:,index]).sum()
                else:
                    importances[index] = len(new_x)-((new_x[:,index].astype(
                                          x[feature].dtype) -
                                          std[index]/sensibility[feature]<
                                          counterfactuals[:,index].astype(
                                          x[feature].dtype)) & (
                                          counterfactuals[:,index].astype(
                                          x[feature].dtype)<
                                          new_x[:,index].astype(
                                          x[feature].dtype) +
                                          std[index]/sensibility[feature])).sum()
        
        results = {col:importances[index] for index, col in enumerate(cols)}
        if print_results:
            
            for k, v in results.items():
                print('Feature {} importance is {}'.format(
                    k, v))
            
            
            
        
        if visualise_results:
            
            plt.bar(np.array(cols), np.array(importances), color = 'red')
            
            plt.title('Importance of Features in the Model\n\
                      (by number of times the feature get changed in a counterfactual)')
            
            plt.xlabel('Feature')
            
            plt.ylabel('Count of Change in Counterfactuals')
            
            if len(cols[np.argmax([len(feat) for feat in cols])]) + len(cols)>30:
                plt.xticks(rotation=90)
            
            plt.show()
        
        return results