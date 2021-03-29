# CERTIFAI
A python implementation of CERTIFAI framework for machine learning models' explainability as discussed in https://www.aies-conference.com/2020/wp-content/papers/099.pdf 

To quote the framework:
Shubham Sharma, Jette Henderson, and Joydeep Ghosh. 2020. CERTIFAI:
A Common Framework to Provide Explanations and Analyse the Fairness
and Robustness of Black-box Models. In Proceedings of the 2020 AAAI/ACM
Conference on AI, Ethics, and Society (AIES ’20), February 7–8, 2020, New
York, NY, USA. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/
3375627.3375812

## Installation
Clone this repository and change the working directory to be inside the cloned repository.
To install the dependencies needed by the repository, either use pip:
```
pip install -r requirements.txt
```
Otherwise you can use anaconda to create a virtual environment and install the libraries there:
```
conda create -n certifai --file requirements.txt
conda activate certifai
```

You're ready to go.


## Background
The CERTIFAI.py module contains the CERTIFAI class, implementing the CERTIFAI framework illustrated in the referenced work. The framework aims at building a model-agnostic interpretation framework, based on the generation of "counterfactuals", i.e. data points that are as close as possible to actual data samples but for which the model output a different prediction. Different analyses over these counterfactuals can then be used for different purposes, as described in more details in the original paper and below in the use cases.

## Usage
### Fitting to a dataset
As a first step, the CERTIFAI class needs to be instantiated. To do so, 3 options are available:
1. Instantiate the class without additional arguments. In this case, the dataset will have to be passed as an argument together with the trained model in all the class methods. (e.g.)
```
certifai_instance = CERTIFAI()
```
2. Instantiate the class using the dataset_path or the numpy_dataset arguments. In the first case, the path to a .csv file can be provided and the instance will be generated so as to have the referenced dataset in the form of a pandas dataframe. In the second case, a numpy array is passed when instantiating and the dataset will be stored in the form of a numpy array. In both cases, the dataset is stored under the attribute "tab_dataset".
(from .csv)
```
certifai_instance = CERTIFAI(dataset_path = 'User/example_path/example_file.csv')
print(type(certifai_instance.tab_dataset))
pandas.DataFrame
```
(from numpy array)
```
example_array = numpy.array([[0,1],[0,1]])
certifai_instance = CERTIFAI(numpy_dataset = example_array)
print(type(certifai_instance.tab_dataset))
numpy.ndarray
```
3. Instantiate the class via the class method *from_csv*, by providing the path to a .csv file. This option works in the same way as passing the .csv path to dataset_path when instantiating.
(e.g.)
```
certifai_instance = CERTIFAI.from_csv('User/example_path/example_file.csv')
print(type(certifai_instance.tab_dataset))
pandas.DataFrame
```

Once the instance is ready, the fit method will generate the counterfactuals for each sample in the referenced dataset, under the given trained model.
(minimal use example)
```
certifai_instance.fit(model)
```
(use if dataset was not provided when instantiating, as described above)
```
certifai_instance.fit(model, x = include_here_your_dataset)
```

Additional options are described in details in the module (some further common use examples will be included shortly). The results of the *fit* method will be stored in the *results* attribute of the class instance and will include a list of tuples each containing the original sample at that index in the dataset, the generated counterfactuals for that sample and distances from the original sample of those counterfactuals.
(e.g.)
```
certifai_instance.fit(model)
print(type(certifai_instance.results))
list
print(type(certifai_instance.results[0]))
tuple
```
N.B. The *results* attribute won't contain anything if the *fit* method is not called.

### Check Robustness
The robustness of the model in the framework is described as the average distance of the counterfactuals from the relative original samples. If the average distance is higher, the model is more robust, because to generate a different prediction a more different data point is needed, while in the opposite case a different prediction can be obtained with less "effort" (see the paper for more details). The CERTIFAI class has a method *check_robustness* that computes the average distance of the counterfactuals and that can be called after the use of *fit* to generate the counterfactuals. Both normalised and unnormalised case is supported (see paper for more details):
(unnormalised robustness)
```
certifai_instance.check_robustness()
```
(normalised robustness)
```
certifai_instance.check_robustness(normalised = True)
```

### Check Fairness
The fairness of the model for specific subgroups of the original dataset can be checked by comparing the robustness of the model for those different subsets. The CERTIFAI class includes a method *check_fairness* that does exactly this. It accepts as an argument the conditions on which the subsets of the dataset needs to be obtained in the form of a list of dictionaries, whereas each dictionary contains the partitioning condition(s). At the moment, just categorical feature in the dataset can be used for partitioning the dataset in this method. The key(s) of the dictionaries need to match the column names of the dataset, while the values need to match a plausible value of that column in the dataset. The method returns the average (un)normalised robustness score for each subset as values in a dictionary having the partitioning conditions as keys.
(e.g.: check fairness of observations for male against female samples in the dataset. Here we assume that the dataset contains a column named 'Sex' that can take at least 'Male' and 'Female' as values)
```
conditions = [{'Sex': 'Male'}, {'Sex': 'Female'}]
certifai_instance.check_fairness(conditions)
```
Multiple conditions can be passed in each dictionary inside the conditioning list and the robustness score can be normalised in the same way as for the *check_robustness* method. Additionally, a visualisation via matplotlib can be generated by setting the argument *visualise_results* to True and the results can be printed more explicitly by setting to True the *print_results* argument.

### Features Importance
The relative importance of the features in the model are obtained by counting the number of times the given feature changed in generating a counterfactual. The paper does not explicitly says how to treat continuous variables under this respect. This counting is performed by the class method *check_feature_importance*. In this implementation, a continuous variable is counted as "changed" if the counterfactual is bigger or smaller than the original value +- standard_deviation/sensibility, whereas the standard deviation is computed for each continuous feature in the dataset and the sensibility is an argument of the described method and, as such, can be chosen by the user (default is 10 and different values for different features can be passed by using a dictionary having features' names as keys and chosen sensibilities as corresponding values). 
(minimal example)
```
certifai_instance.check_feature_importance()
```
(Setting the sensibility, i.e. same for every continuous feature)
```
certifai_instance.check_feature_importance(sensibility = 5)
```
(Setting a different sensibility for each continuous feature)
```
certifai_instance.check_feature_importance(sensibility = {'continuous_feature_name_1': 5, 'continuous_feature_name_2': 10})
```
Similarly to the *check_fairness* method, the results can be visualised via the *visualise_results* argument being set  to True.

## Examples
An example of the use of CERTIFAI class is included in ModelExCERTIFAI.py. The example script includes also the training of a classification model on the drug200 dataset with PyTorch and PyLightning. To run the script just run it with python after having installed all dependencies.
