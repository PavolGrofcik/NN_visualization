# NN_visualization

*A simple animation of the learning three-layered regression Neural Network*

 
![NN_visualization](https://github.com/PavolGrofcik/NN_visualization/blob/main/NN_visualization_new.gif)

### NN parameters
Neural network acting in background as model was implemented from scratch to understand & learn logic in NN on my own.  

**Layers:** 3  
**Inputs:** 3  

**ACF Hidden Layer:** ReLu  
**ACF Output Layers:** Identity (1)  
**Loss:** MSE, in animation 1 sample => SE   
**Learning rate:** 0.0007   
**Epochs:** 25  
**Dataset:** [QSAR fish toxicity Data Set](https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity)  

**Notice:**  
In the hidden layer there are 2 neurons with its potential equal 0. Since its potential is 0, and 
ReLu is not diferentiable in a point 0, the process of learning (updating weights) does not have
any effect on them. In order to avoid this trade-off, there are normalization techniques such as 
*Dropout* or preffered other ReLu variants - Leaky ReLu, Elu, etc... 
  
  

  
**Implemented:** Python  
**GUI:** PyQt5 
