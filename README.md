This is an implementation of a CNN and KNN to analyze and classify dermoscopic images, in the different categories.
To run and implement it is necessary to download the dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
Follow the next steps for running the KNN code:                                                                                  
  1. Download the data set.                                                                                                                       
  2. Run the file histograms.py with the changes in the direction where the data set is downloaded this will create histograms of the channels
  of the RBG palette of colors.
  3. Run the file PCA.py - This will reduce the number of dimensions to process the information more easily. 
  4. Run CrossValidation.py - Create a Test, Training, and validation information set for the later KNN.
  5. Run the KNN.py - This will print the precision of the test and classify it in confusion matrices, F score, and Accuracy.
                                                                                                                                
For the CNN is not necessary to pre-train the model, because the CNN works directly with images, so just run it 
