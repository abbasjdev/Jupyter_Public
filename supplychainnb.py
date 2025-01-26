##This file only contains the python code from the Supply Chain analysis

import os #For operating System tayloring
import kagglehub #For using library to download datasets directly from Kaggle API
import pandas as pd #Commonly used library to analayze and visualize datasets
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt



# Download latest version
path = kagglehub.dataset_download("amirmotefaker/supply-chain-dataset")

print("Path to dataset files:", path)

sv_file_path = os.path.join(path, "supply_chain_data.csv")  
print("Full file path:", csv_file_path)



if os.path.exists(csv_file_path):  # os.path.exists requires os
    # Load the CSV into a pandas DataFrame
    data = pd.read_csv(csv_file_path, index_col=False, header=0)
    print(data.head())  # Display the first 5 rows
else:
    print("The file was not found at the specified path.")
    
    print("Full file path:", csv_file_path)
    
    
    
num_rows = len(data)


print("The total number of rows in this dataset is: ",num_rows)


df = data
 
y = pd.to_numeric(df['Number of products sold'], errors='coerce')
X = df.drop(columns=['Number of products sold']).select_dtypes(include=['number'])

# Add a constant to the independent variables (for the intercept)
X = sm.add_constant(X) 


model = sm.OLS(y, X).fit()


p_values = [model.pvalues]

print(sorted(p_values))


x = df['Production volumes']
# Make predictions
predictions = model.predict(X)

#We need to make sure the shape of each variable matches to prevent errors. 

print(x.shape) 
print(y.shape)

# Plot the data and the regression line
plt.scatter(x, y, label='Data points', color='blue')
plt.plot(x, predictions, color='red', label='OLS fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('OLS Regression Line')
plt.show()
