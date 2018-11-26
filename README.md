# Traffic Sign Classifier

Traffic sign classifier using a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network).

### Model Architecture

| Layer         				| Description	 	       		    | 
|:------------------------------|:----------------------------------| 
| Input         				| 32x32x1 grayscale image   		| 
| Convolution 5x5 (L1)    		| 1x1 stride, outputs 28x28x32 		|
| RELU							|									|
| Max pooling 2x2     			| 2x2 stride, outputs 14x14x32		|
| Convolution 5x5 (L2)	    	| 1x1 stride, outputs 10x10x64    	|
| RELU 							|									|
| Max pooling 2x2				| 2x2 stride, outputs 5x5x64		|
| Convolution 3x3 (L3)			| 1x1 stride, outputs 3x3x128		|
| RELU 							|									|
| Max pooling 2x2				| 1x1 stride, outputs 2x2x128		|
| Inception Module:				|									|
| - Max pooling 4x4 (L1)        | 2x2 stride, outputs 6x6x32		|
| - Max pooling 2x2 (L2)		| 2x2 stride, outputs 2x2x64		|
| - Flatten (L1)				| outputs 1152						|
| - Flatten (L2)				| outputs 256						|
| - Flatten (L3)				| outputs 512						|
| - Concat (L1 + L2 + L3)		| outputs 1920						|
| - Dropout 					|									|
| Fully connected				| outputs 800   					|
| Softmax						| outputs 43      					|