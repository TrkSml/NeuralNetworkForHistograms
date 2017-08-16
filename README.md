# NN for histograms


![nn](https://user-images.githubusercontent.com/21022345/29356110-749fed92-8273-11e7-8a9e-1ab0592180c2.png)


	Input features size = 111
	Hidden Layers = 1 X 55

# Classifying histograms with a Neural Network.  

Histograms are represented as numpy arrays where each elem is number of Points in each [Yi, Yi + dy] (or each bin). We have 111 Features and one output (Binary Classification).


Why Kears ? 
	* Sklearn MLPClassifier module is poor compared to APIs like Theano. 
	* Keras is well documented and very user friendly. In fact, if you are already familiar with sklearn classifiers, you wont be disorientated.
	* Keras allows you to save models and load them. Models are stored as JSON and YAML files which is perfect. 
	* Keras exemples in officiel GitHub are amazing and very helpful. 




