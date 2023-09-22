import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def plot_classification(model, X, y, col1, col2):
    '''
    Plot the classification boundary of a model
    '''
    # grid data
    min_x = X[col1].min()
    max_x = X[col1].max()
    min_y = X[col2].min()
    max_y = X[col2].max()
    x1 = np.linspace(min_x, max_x, 100)
    y1 = np.linspace(min_y, max_y, 100)
    xx, yy = np.meshgrid(x1, y1)
    inp_df = pd.DataFrame({
        col1: xx.flatten(),
        col2: yy.flatten()
    })
    # predict
    Z = model.predict(inp_df)
    Z = Z.reshape(xx.shape)
    # plot
    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, Z, cmap='cool', alpha=0.5)
    plt.scatter(X[col1], X[col2], c=y, cmap='hot')
    plt.xlabel(col1)
    plt.ylabel(col2)
    return plt