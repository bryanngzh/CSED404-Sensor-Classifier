from libsvm.svm import *
from libsvm.svmutil import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Step 1: Load dataset from a single file
def load_data(filename):
    labels = []
    features = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])  # First value is the label
            feature_dict = {int(f.split(":")[0]): float(f.split(":")[1]) for f in parts[1:]}  # Parse features
            labels.append(label)
            features.append(feature_dict)
    return labels, features

# Step 2: Train and evaluate the SVM model
def train_and_evaluate(data_file, model_file):
    # Load the dataset
    labels, features = load_data(data_file)

    # Split the dataset into 80% training and 20% testing
    train_labels, test_labels, train_features, test_features = train_test_split(
        labels, features, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train the model
    print("Training the model...")
    prob = svm_problem(train_labels, train_features)
    param = svm_parameter('-t 2 -c 1 -g 0.5')  # RBF kernel, C=1, gamma=0.5
    model = svm_train(prob, param)
    
    # Save the trained model
    svm_save_model(model_file, model)
    print(f"Model saved as {model_file}")
    

    # Evaluate the model
    print("Evaluating the model...")
    predicted_labels, accuracy, _ = svm_predict(test_labels, test_features, model)    
    print(f"Accuracy: {accuracy}%")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    print(cm)

    return model

# Main Execution
if __name__ == "__main__":
    data_file = "data/data.txt"  
    model_file = "activity_model.model"

    # Train and evaluate the model
    train_and_evaluate(data_file, model_file)
