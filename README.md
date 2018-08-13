# Mimetik

Mimetik is a machine learning software.  
Main features:
- Creation of a neural network (multiLayer perceptron) 
- Load Training data set from a file
- Learning
- Save neural network state
- Compute new inputs
- Execute script

You can use it as a command line program:

    network 2 5 1
    loadTrainingSet xor.txt
    learning 5000
    saveState save_xor
    compute 0 0
    compute 0 1
    compute 1 0
    compute 1 1
    
Or a c++ library (multilayerPerceptron.h):

    vector<int> layers;
    layers.resize(3);
    layers[0] = 2;
    layers[1] = 5;
    layers[2] = 1;
    
    multilayerPerceptron network(layers);
    network.learning("./xor.txt", 5000);
    network.saveState("save_xor");
    network.computeFile("./xor.txt");
	
## Installation
    make 
    make install
## Execution
Start mimetik :

    mimetik

Run a mimetik script:

    mimetik script.mimetik

## Command line

    usage:
    	network nbLayer1 nbLayer2 ... - Create neural network layers
    	loadTrainingSet trainingset.txt - Load training set from file
    	setEta eta - set learning rate factor [0,1] (default = 0.5)
    	setAlpha alpha - set momentum factor [0,1] (default = 0.9)
    	learning limit verbose(booleen) randomOrder(booleen) - Start learning
    	compute input1 input2 ... - compute outputs
    	computeFile fileIn fileOut - compute a file
    	saveState filename - Save neural network state in binary file
    	saveStateText filename.txt - Load neural network state in text file
    	loadState filename - Load neural network state from binary file
    	loadStateText filename.txt - Load neural network state from text file
    	execute script.mimetik - Execute mimetik script
    	exit - Quit the software
    
    examples:
    	network 2 10 5 1
    	loadTrainingSet trainingset.txt
    	setEta 0.5
    	setAlpha 0.9
    	learning 5000 true false (or: learning 5000)
    	compute 0.5 0.1
    	computeFile fileIn.txt
    	saveState weights.bin
    	saveStateText weights.txt
    	loadState weights.bin
    	loadStateText weights.txt
    	execute script.mimetik

## Files
### Training set (loadTrainingSet)
Training set files  contain inputs and outputs to allow the neural network to learn by example  
xor: 

    [mlp]
    2 1 4
    
    [inputs]
    0 0
    0 1
    1 0
    1 1
    
    [outputs]
    0
    1
    1
    0
 
  
The header [mlp] defines:
The number of inputs, the number of outputs, the number of examples  
[inputs] are Float values (scaled in [0;1] for better results)  
[outputs] are Float values

### SaveStateText Files

    [mlp_layers]
    3 2 5 1
    
    [mlp_weights]
    -2.39106 -2.11835 
    -5.64354 -5.69497 
    0.626959 0.60677 
    -7.0111 3.14889 
    3.16021 -7.02828 
    
    -2.49848 -14.4462 -4.83591 7.35598 7.31272 
   

The header [mlp_layers] defines:  
The number of layers, the number neurons of layer1, the number neurons of layer2 ...  
[mlp_weights] defines weights of the neural network layer by layer

### SaveState Files
Same information as SaveStateText but in binary format (less disk space)

## Use cases 

The folder "examples" contains some use cases.

- xor: Learning xor
- sin: Learning sinus
- cross_circle: Image recognition (cross or circle)
- vehicle: Prediction of a transport means (see vehicle_readme.txt)

### Simple perceptron
In addition, mimetik provides a simple implementation of a perceptron: perceptron.h  
See the perceptron folder
