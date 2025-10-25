# Learn More from YouTube: https://www.youtube.com/watch?v=V_xro1bcAuA
# https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises
# Applying and ploting Linear Regression model
# Import PyTorch
import torch
import torch.nn as nn
# importing matplot
import matplotlib.pyplot as plt
# import matplotlib

# Displays the back-application used for graphing
# print(f"Current backend: {matplotlib.get_backend()}") 

# If default Agg download Tkinter and apply:
# matplotlib.use('TkAgg')

# Creating Known parameter
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])

print(f"Length of X:{len(X)}")
print(f"Length of y:{len(y)}")

# Create train split or test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# Creating Visual presentation


def plot_prediction(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):
    # Plots training data, test data and compares predictions.
    plt.figure(figsize=(10, 7))
    # Plotting Training data in Blue Color
    plt.scatter(train_data, train_labels, color="#0000FF", label="Train")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Linear Slope")
    # Plot test data in green
    plt.scatter(test_data, test_labels, color='#FF0000', label="Test")
    # print("Testing and Training figure been saved")
    # Are there prediction
    if predictions is not None:
        # plot the predictions if there exist
        plt.scatter(test_data, predictions,
                    color='#aa0aaf', label="Prediction")
        # plt.savefig("Prediction.png")
        # print("Prediction figure been saved")
    # Show the legend
    plt.legend(prop={"size": 14})
    plt.title("Plot with Legend")
    plt.show()
    # plt.savefig("figure.png") could be used to store the output figure physical form


plot_prediction()
# print("Successfully Complete")



class LinearRegressionModel(nn.Module): # <-- almost everything in PyTorch inhits from nn.module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(
            1,
            requires_grad=True,
            dtype=torch.float
        ))
    # Forward method to define the computation in the model
    def forward(self,x:torch.Tensor) -> torch.Tensor :
            return self.weights * x + self.bias # this is the linear regression formula
        

# Implementation of Random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.module)
model_0 = LinearRegressionModel()

# Check out the parameters
print(list(model_0.parameters()))

# List named parameters
print(model_0.state_dict())

# Making prediction using `torch.inference_mode()`
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)

plot_prediction(predictions=y_preds)

# Setup a loss funtion 
loss_fn = nn.L1Loss()

print(loss_fn)
