# Hands_on_ML

# Follow this
https://cuddly-adventure-wr7qv44pqg4g25pxw.github.dev/

https://github.com/codebasics/nlp-tutorials/blob/main/1_regex/regex_for_information_extraction.ipynb



# To remember

ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough') //town is the name of the colum [0] is the index to which oneHot needs to be applied and passthrough means rest of the columns need to just passed

if the dataset is [1,2,3,4,5,6,7,8,9] and n_splits=3
means the dataset is divided into 3 equal folds which is Total elements/n_splits = 9/3 that is 3
In the first iteration, the first 3 elements are the test set, and the rest are the training set.
In the second iteration, the next 3 elements are the test set, and the rest are the training set.
In the third iteration, the last 3 elements are the test set, and the rest are the training set.


# KFold and StratifiedKFold
KFold is sufficient in case of balanced dataset
Stratified is useful for inbalanced dataset
Suppose you have a dataset with 10 samples, and their target labels are [0, 0, 0, 1, 1, 1, 1, 1, 0, 0].
Using KFold (with n_splits=2), one split could assign more 0s in the test set and fewer 1s, leading to an unbalanced test set.
StratifiedKFold ensures that the proportion of labels (e.g., 0s and 1s) in the training and test sets is approximately the same as in the original dataset.

#Cross Validation 
Here n_estimators is the number of decision trees whereas cv defines the number of folds . It uses stratified Kfold by default

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
Here 0 represents the first column which represents age feature and 1 represent income feature(select all this features)


#Elbow Plot
An elbow plot is a graphical tool used to determine the optimal number of clusters in clustering algorithms, particularly in K-means clustering.
The elbow plot plots:

X-axis: The number of clusters (k).
Y-axis: A metric that measures clustering performance, commonly the within-cluster sum of squares (WCSS) or inertia, which represents the total distance of points from their respective cluster centroids.

dataset.data[0].reshape(8,8)// reshapes 1D array into 2D array


