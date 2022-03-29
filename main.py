from feedforwardneuralnetwork import FeedForwardNeuralNetwork
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    iris = datasets.load_iris()
    encoded_y = []
    for y in iris.target:
        if y == 0:
            encoded_y.append([1, 0, 0])
        elif y == 1:
            encoded_y.append([0, 1, 0])
        else:
            encoded_y.append([0, 0, 1])
    X_train, X_test, y_train, y_test = train_test_split(iris.data, encoded_y, test_size=0.2)
    standard_scaler = StandardScaler()
    X_train_normalized = standard_scaler.fit_transform(X_train)
    X_test_normalized = standard_scaler.fit_transform(X_test)
    neural_net = FeedForwardNeuralNetwork()
    neural_net.create_layer(12, "relu", 4)
    neural_net.add_layer(3, "sigmoid")
    neural_net.train(X_train_normalized, y_train, 10, learning_rate=1)
    for i in range(len(X_test_normalized)):
        O = neural_net.predict(X_test_normalized[i])
        print("Prediction: " + str(O))
        print("Actual: " + str(y_test[i]))
    neural_net.export_model("sigmoidXsigmoid.json")