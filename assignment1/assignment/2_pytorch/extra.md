Name: Reagan Kan
Email ID: rkan3
Best Test Accuracy: 65% (from run_mymodel.py)
Best Test Accuracy: 67% (from run_test.py and submitting predictions.csv)
______________
Things I tried: I followed the suggestions in the HW instructions.
______________
Filter size: Tried the following: 7x7, 5x5, 3x3. Given enough training, all filter sizes reached roughly the same testing accuracy of about 60-65%. The 3x3 filter was most efficient in the sense that it converged to that accuracy the quickest. Thus, the final model that I used for the contest had a kernel size of 3.
______________
Number of Filters: I tried filter counts of 10, 32, 64, with testing accuracies of 58%, 62-3%, 62-3%. Raising the filter count helps grow the model class and reduce modeling error. Past a certain point, the complexity of the model is enough to solve the problem. This is evident in the lack of testing accuracy improvement between 32 and 64 filters. Since performance was similar, the final contest model has 32 filters to maintain a less complex model and avoid overfitting.
______________
Network Depth: I  tried three network architectures with the general form of: [conv-relu-pool]x(N) - [affine]x(M) - [softmax or SVM]
network1: [conv-relu-pool]x(1) - [affine]x(1) - [softmax]
network2: [conv-relu-pool]x(2) - [affine]x(1) - [softmax]
The corresponding testing accuracies were 62% and 65%. 

Since 2 conv sandwich layers worked better, I tried this model.
network3: [conv-relu-pool]x(2) - [affine]x(1) - [SVM]
This had 60% accuracy. So, my final contest model uses the structure of network2.
______________
Alternative Update Methods: When trying the various configurations for the above hyperparams, I trained the model using SGD. Then, I took my "final/best" model and re-trained with AdaDelta(32%), AdaGrad(57%), and Adam(59%). My submission to the contest used the SGD(65%) trained version of my final model.