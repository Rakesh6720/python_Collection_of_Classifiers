<p class=font-size: 9px># python_Collection_of_Classifiers</p>
<h2>Python program testing 4 different Sci-Kit Classifiers on a data set.</h2>

In this project I've taken an arbitrary dataset of a group of people's measurements, put them in a 4 x 3 matrix of Vector3 values, and created a predictive model based on gender attribute.  The 5 classifiers used are:

<h3>
<b>Decision Tree</b>
</h3>
Supervised, greedy, heuristic model using the ID3 algorithm to separate elements of a set by decreasing amounts of entropy.  Similar members of class form subsets in leaves edges of which determine the rules of classification.  

<h3>
<b>K Nearest Neighbors</b>
</h3>
Determines the class of an element based on its proximity in value to other members of the class.

<h3>
<b>Support Vector Machines</b>
</h3>
A supervised learning model that uses training data to classify examples to categories using linear classification in vector space.  I speculate this model could be prone to overfitting on a limited data set.  

<h3><b>Multilayer Perceptron</h3></b>
Supervised learning that uses backpropagation to fix errors as it trains on data.  Stochastic Gradient DEscent and Backpropagation are the de facto models for neural networks.  
*Surprisingly prone to overfitting at an alpha value below 5.  I speculate because the data set was so small having heigher value for weights as alpha parameter in the MLP method the result skewed.  This was the only classifier to incorrectly classify the test data.  

<h3><b>AdaBoost</h3></b>
Fits a classifier to a data set repeatedly, honing in on the hard to crack problems.  Tweaks "weak learners" into combined weighted sum that reflects a BOOST.  

<h3><b>Quadratic Discriminant Analysis</h3></b>
Classifier with quadratic decision boundary generated by fitting class conditional densities to the data and using Bayes' Rule.  
