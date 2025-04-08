###################################### Decoding #############################################

Decoding is an attempt to 'revert' the encoding process, through which data is transformed into representations in lower-dimensional spaces, aiming to capture characteristics that are not observable when navigating through (hyper)dimensional planes. Specifically, we aim to represent how a mouse maps its neural pathway through space during a given task, highlighting hidden structures that are not immediately apparent.
Decoding, as anticipated, is somewhat of an inverse process. Starting from the aforementioned representation, it attempts to reconstruct the actual spatial path taken by the mouse.

   The process, which is the same for all the methods used (and is tun 30 times each) is organized into several steps:


   1) Initial split of the database into training and test sets (0.7-0.3). The training set is further divided into sub-train and validation, which w
    	will be used to tune the decoder. (Specifically a KNN) (0.85-0.15)

   2) We fit the chosen model on the initial training set; then we apply the transform to the validation set, the internal training set, and the initial test set.

   3)  We use the validation set to tune the decoder, specifically to find the best k-value (which will obviously be the one that minimizes the 
	discrepancy between predicted values, given the embedding, and observed values, such as position and direction).


   4) Once we have the best k-value for the given round, we fit the decoder on the training data (the embedding we get from transforming the training  
    data along with the training data itself). Then, using the fitted decoder, we make predictions on the test embedding (as described in step 2).

   5) I calculate three metrics: two R² values comparing the observed position and direction with the predicted ones, and a position error, which is  
    the absolute difference between the predicted and observed positions.


   6)  After collecting all the results, I create a box plot to compare the median prediction error across different methods.

   7) Subsequently, I run some ANOVA tests to check for any significant differences between the methods and the pairwise Tukey's test to minimize type 
    I error






