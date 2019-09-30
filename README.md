# VIC-on-QS-Rank
Machine learning project where we implement 10-fold cross-validation with 5 models:
- Logistic Regression
- Random Forest
- SVM
- LDA
- Naive Bayes
## Getting Started

To deploy the code, run every cell of the vicQSrank.ipynb 

### Prerequisites

What things you need to install the software and how to install them

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
```

### Installing

A step by step series of examples that tell you how to get a development env running

Run the following five method functions:

```
svmFunct(X,y)
randomForestFunct(X,y)
logregFunct(X,y)
ldaFunct(X,y)
naiveBayesFunct(X,y)

VIC Implementation cell 
```

And two dictionaries are stored after running this cells.

```
bestAveragesScores
bestMaxScore
```

The bestAveragesScores dictionary stores the 5 functions averages of the AUC per data base.
bestMaxScore stores the name of the best model and the AUC of that model

## Running the tests

Each classifier has its corresponding function, the VIC implementation calls the 5 function's model and stores the results of each model as a dictionary.

### Break down into end to end tests

A run of one fold of every function returns the following values.

```
[[16  0]
 [ 4  1]]
AUC: 0.6
[[14  0]
 [ 4  2]]
AUC: 0.6666666666666666
[[11  2]
 [ 6  1]]
AUC: 0.49450549450549447
[[16  0]
 [ 2  2]]
AUC: 0.75
[[11  1]
 [ 7  1]]
AUC: 0.5208333333333334
[[14  0]
 [ 5  1]]
AUC: 0.5833333333333334
[[12  3]
 [ 1  4]]
AUC: 0.8
[[15  0]
 [ 4  1]]
AUC: 0.6
[[15  2]
 [ 2  1]]
AUC: 0.6078431372549019
[[12  4]
 [ 3  1]]
AUC: 0.5
[ 136   12 ]
[ 38   15 ]
10-FOLD AVERAGE OF SVM IS: 0.6009688934217237
```

### And coding style tests

Three partitions of bestAveragesScores sample:

```
{'/content/drive/My Drive/datasets/dataset_100.csv': {'LDA': 0.760940594059406,
  'Logistic Regression': 0.8603960396039605,
  'MAX Score': ('Random Forest', 0.8704455445544554),
  'Naive Bayes': 0.8303465346534653,
  'Random Forest': 0.8704455445544554,
  'SVM': 0.8501485148514851},
 '/content/drive/My Drive/datasets/dataset_101.csv': {'LDA': 0.7714851485148515,
  'Logistic Regression': 0.855990099009901,
  'MAX Score': ('Random Forest', 0.8658910891089109),
  'Naive Bayes': 0.8263861386138613,
  'Random Forest': 0.8658910891089109,
  'SVM': 0.8463366336633663},
 '/content/drive/My Drive/datasets/dataset_102.csv': {'LDA': 0.7918894830659537,
  'Logistic Regression': 0.85650623885918,
  'MAX Score': ('Random Forest', 0.8612596553773024),
  'Naive Bayes': 0.8324420677361855,
  'Random Forest': 0.8612596553773024,
  'SVM': 0.8523469994058229}}
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Sci-kit](https://scikit-learn.org/stable/) - The framework used for the models
* [Pandas](https://pandas.pydata.org) - DataFrame framework
* [Matplotlib](https://matplotlib.org) - Library used for plots

## Authors

* **Ana Estrada** - (https://github.com/AnaCReal)
* **Emilio Ferreira** - https://github.com/efferreiram)
* **Rodrigo Careaga** - (https://github.com/lrodrigocareaga)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
