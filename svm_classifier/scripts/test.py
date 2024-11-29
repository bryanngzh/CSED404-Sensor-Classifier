import numpy as np
import scipy.sparse as sp
from libsvm.svm import *
from libsvm.svmutil import *

# Assuming the model is already loaded
def load_model(model_file):
    """Load SVM model from file"""
    return svm_load_model(model_file)

# Load your pre-trained model
model = load_model("activity_model.model")

# Define your node data (index, value s)
nodes = [
    (1, 0.7425617057166223),
    (2, -0.9998375745690188),
    (3, 0.3019591408357476),
    (4, -0.9999709490763508),
    (5, 0.7734776211222758),
    (6, -0.9999583696974764),
    (7, -0.6451518763682863),
    (8, -1.0000005237434908),
    (9, 1.0358830031955444),
    (10, -0.9999995770404488),
    (11, 2.8329457622845804),
    (12, -1.0000009353338188),
    (13, 0.7387022766339866),
    (14, -1.0000004541914658),
    (15, 0.3040077030762971),
    (16, -0.999999508612014),
    (17, 0.7810255567926319),
    (18, -0.9999992461785141)
]

# Prepare the data for prediction in libsvm format (dense or sparse)
# Let's assume it's a dense vector first
def prepare_dense_vector(nodes):
    # Create a list with 18 elements, one for each feature (index 1 to 18)
    vector = np.zeros(18)
    
    for index, value in nodes:
        vector[index - 1] = value  # Adjust for 0-based indexing in numpy
    
    return vector

# Prepare the input features vector for prediction
X_predict = prepare_dense_vector(nodes)

# Convert to the format expected by libsvm (like [1: 0.433854, 2: -0.10405, ...])
# Now use svm_predict for the model
labels, _, _ = svm_predict([0], [X_predict], model)

# Print the prediction
print(f"Predicted label: {labels[0]}")
