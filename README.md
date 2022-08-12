# CNT-Epilepsy-Patient-State-Clustering
This will host the files used for the project of cleaning &amp; preprocessing data from wearable devices of epilepsy patients in order to then create &amp; train unsupervised clustering machine learning models for the detection of the different medical states an EMU patient is in (NOTE: all code for this project was performed on Python). This project had limited data, only accessing the data from a few patients. Do the the IRB approval, none of the data itself is allowed to be shared. The data consisted of mainly the following recorded dignals: ECG heart rate, Apple Watch heart rate, and Apple Watch accelerometer data. The availability of each of these data types varied from patient to patient. Using these signals, the goal was to be able to detect the different states an EMU epilepsy patient is in through a multi-day time period. 

The first step in this was the process of cleaning and preprocessing the data. This involved tackling the issue of varying sampling rates among the devices used to record the signals. Data reduction and smoothing was accomplished by performing a moving average of the signals, usually at about 30 second moving averages. This allowed the resulting "sampling rate" after performing the moving averages to be equal among all signal DataFrames. These 30 second moving averages also functioned as (non-overlapping) sliding windows in the data to be used later in the process of training clustering models. One additional step was necessary in order for the data for all signals to properly align/match in time throughout the entire data set. Different signal DataFrames were merged together and cut towards the beginning and/or end (depending on which was necessary) in order for the start times and end times for the recordings of all signals to match. 
Specifically when dealing with patients with Apple Watch accelerometer recordings, additional data reduction was necessary; the accelerometer recordings tended to have much higher sampling rates compared to the ECG & watch heart rate recordings. Moving average signal smoothing was performed individually on the Apple Watch accelerometer data (1 second moving average) prior to performing the 30 second moving averages on all DataFrames of the dataset. 
This entire preprocessing procedure was made into an efficient and quick-to-use function within a class, allowing other researches to efficiently perform all data reduction, cleaning, and beneficial preprocessing steps when dealing with all this clinical data. 

With the preprocessed data, this already created the begininning of the DataFrame for the feature set to be used in the training of clustering algorithms. However, the next step was to increase the size of this feature set through feature selection. Adding onto the already created class and function described above for preprocessing, features were created involving ratios of already existing features, derivatives (involving Fourier Transforms), etc. A goal for the future of this project is to add on additional features, which can easily be done by simply adding on the code for this to the already existing file for preprocessing and feature selection. 

With the click of a button (using this file), one can perform all preprocessing and feature selection for multiple signal recordings and multiple patients, all at once. After doing so, the next steps taken were checking for any strong correlation among features, which is done in order to avoid multicollinearity. If this problem is found, to options can be taken: either deleting one of the highly correlated features, or better yet, performing PCA at a number of principal components that maintains as much variance as possible while simultaneously avoiding multicollinearity. In the example performed in the Jupyter notebook, a feature was deleted. However, in the future, the goal is to create a Pipeline of the class/functions created above, along with performing PCA afterwards, with the number of components as "mle", which will help maintain variance. After analysis of the features, scaling is performed with StandardScaler(). This is another step that would like to be added to the Pipeline mentioned above, in which standardization/scaling would be performed prior to PCA. At this point, the data is ready to be applied for the training of various clustering models. 

Several unsupervised clustering models were attempted, including KMeans, Agglomerative Clustering (a type of hierarchical clustering), and Gaussian Mixture Models. For KMeans models, silhouette scores were used as a method of determining the optimal number of clusters to use, while agglomerative clustering on Python answered that question for you with the visual display of a dendrogram. For Gaussian Mixture Models, Bayesian Information Criterion (BIC) and their gradients were mainly used as a method of determing the number of clusters, which is preferred over silhouette scores due to the more complex nature of clustering performed with GMM models. 

Lastly, several methods (adjusted random scores &amp; adjusted mutual information scores) were used in order to check for similarity among the created clustering models and see which ones, if any, resulted in similar clusters. With these scores, it was proven that the results of clustering were consistently different among KMeans models & Gaussian Mixture Models (GMM). This makes reasonable sense due to the difference in clustering techniques taken by the two. GMM tends to use a more soft clustering technique, allowing overlaps in the clusters, while KMeans has more strict cutoffs between clusters. GMM also has the ability to detect more complex patterns that can be hidden for KMeans models.

The resulting clusters were then analyzed over time to see if there were any patterns over time for certain clusters. Additionally, noted seizure times were plotted along with the clusters, which also allows us to see if the presence (or absence) of certain clusters can be associated with the occurence of a seizure or the preictal period of a seizure. Additionally, one fo the next steps that would like to be taken is to also plot the times in which the patient takes their medication, which will also allow us to see how the medication affects their medical state. This analysis of the differnet clusters (and their attributes) with respect to time will help as part of a bigger goal in moving towards the ability to detect seizures ahead of time by the use of signal data collected from wearable devices, such as an Apple Watch. 
