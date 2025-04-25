# AI Class Final

**Project Definition:** Creating a AI that can analyze text and say if it is a positive, neutral, or negative statement. Such as reviews where a buyer might say something positive about the product, etc. I use pytorch in the project.

**Input:** Short text

**Output:** Positive, neutral, or negative values.

**Model Architecture Overview:** Bidirectional LSTM, Embeddings, hidden layers, and Dropout regularization

Further Model Architecture:
- Embeddings: Convert token IDs into dense vector represenations of them
- BiDirectional LSTM: Processes sequences in both directions.
- Fully Connected Layer: Maps to the final classes of negative, neutral, and positive.

Data Used:
- https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset

My Process:

First i researched on different ways to implement the model and found that LSTM seemed to be a good start so I implemented a basic version of it and it worked. However, when training I found that the model seemed unable to go further than a 70% accuracy. I then asked AI to help me improve it and did soem research and implemented a bidirectional lstm mode, but that trained faster but the accuracy still couldn't break 70%. AI helped with a lot of the orginization of the code as it was messy before.

Update:

I have compiled together multiple datasets for 1 large datasets with about 2.5 million unique entries. After training a simple model for 1 epoch which took about an hour I got much higher accuracy for negative and positive but the neutral got much worse. I believe this is because there are not as many neutral rows and positive or negative. I am going to train the model even more and possibly adjust the model further. I want to atleast train 1 model for about 5 epochs if not more. If that fails to improve the accuracy of the neutral class then I will attempt to implement other methods to get it.

Update 2:

I have recently got the program to detect and use the directml backed but when attempting to train the model fails. I believe it has something to do with the LSTM layer. This means that I won't be able to train the model since training on the cpu will take multiple days and possibly weeks of training. 

If somebody is able to fix it please let me know so I can implement it and push it to the repo. For now this will sit here unable to be trained but if somebody wants to train it themselves then I am using a conda enviroment with python version 3.11. You just have to have a gpu and install the needed dependecies and you will be set and you can run main.py. You can use the dataset that I compiled together on the github. It is in the releases section.

