# MLutilities
This python library aims to provide modules that can be useful to teach Data Analysis and Machine Learning.



Installation: to install this package simple use the following command
```
pip install mlutilities-udea
```
# Basi Usage
Using the mlutilities library for Exploratory Data Analysis (EDA)

### Univariant Analysis
In this example, we demonstrate how to use the mlutilities library to load a dataset, perform the Kolmogorov-Smirnov goodness-of-fit test, and visualize the data.

```python
from mlutilities.datasets import load_dataset
from mlutilities.eda import kolmogorov_test

# First, we load the "penguins" dataset into a Pandas DataFrame.
data = load_dataset(data_set="penguins", load_as="dataframe", n=-1)

# We print the description of the dataset to provide some information about it.
print(data["DESC"])

# We display an image associated with the dataset.
display(data['image'])

# Next, we extract the data from the dataset for further analysis.
df = data["data"]

# We perform the Kolmogorov-Smirnov test on the "bill_depth_mm" variable and plot its histogram.
kolmogorov_test(dataset=df, variable="bill_depth_mm", plot_histogram=True)
```

You can find more example on the [notebooks](./notebooks/) folder.
