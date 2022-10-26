# CNT-Epilepsy-Patient-State-Clustering

This project focused on the cleaning and preprocessing of data from wearable devices of epilepsy patients, along with feature extraction, in order to then create &amp; train unsupervised clustering machine learning models for the detection of the different medical states an EMU patient is in. This project had limited data, only accessing the data from a few patients. Due to the IRB approval, none of the data itself is allowed to be shared. The data consisted of mainly the following recorded signals: ECG heart rate, Apple Watch heart rate, and Apple Watch accelerometer data. The availability of each of these data types varied from patient to patient. Using these signals, the goal was to be able to detect the different states an EMU epilepsy patient is in throughout a multi-day time period. This was my first big project since joining as a researcher at the Center of Neuroengineering and Therapeutics (CNT) under Dr. Brian Litt. 

NOTE: all code for this project was performed in Python.

## Background:

One of the big-picture goals at CNT is to be able to use obtained data from wearable devices in order to to predict when a seizure may occur (ahead of time) for epilepsy patients. This involves training algorithms with pre-ictal data (before the ictal stage of seizure) from wearable devices, such as an Apple Watch. A preliminary step in the process involves obtaining a better understanding behind the different experienced symptoms and medical states an epilepsy patient is in throughout a multi-day period. This can help to better understand what states are associated with the pre-ictal phase, ictal phase, and times completely unrelated to seizure activity. Therefore, this project focused on creating and training unsupervised learning clustering algorithms that are able to detect the different states a patient is in during their multi-day stay at an EMU. The data used for this project included ECG data, heart rate data from an Apple Watch, and accelerometry data from an Apple Watch. Not all patients had every one of these signals recorded; creation and training of these clustering algorithms were patient-specific (i.e. a new and unique algorithm was created and trained for each subject). It is important to note that one of the largest obstacles behind this project was the limited data; data was only acquired from less than a handful of patients. 

## Methods:

### Cleaning and Pre-processing:

The first step in this centered on cleaning and preprocessing the data. This involved tackling the issue of the varying sampling rates among devices used to record the signals. Apple Watch recordings, for example, did not have a constant sampling rate. The sampling rate of the watch would change with respect to the data recorded over time. The ECG signal, although it had a constant sampling rate, did not match the sampling rate of the other devices. Additionally, devices (including both the Apple Watch and the ECG) were often disconnected for varying reasons (such as low battery and necessary charging). This would cause time gaps in the recorded signals as well, which needed to be excluded from the data set so that all recorded signals matched in time (i.e. so that all signals are included simultaneously in all the used time periods of the data). Data reduction and smoothing was accomplished by performing a moving average of the signals, usually at about 30 second moving averages. This allowed the resulting "sampling rate" (after performing the moving averages) to be equal among all signal DataFrames. These 30 second moving averages also functioned as (non-overlapping) sliding windows in the data to be used later in the process of training clustering models. One additional step was necessary in order for the data of all signals to properly align/match in time throughout the entire data set. Different signal DataFrames were merged together and cut towards the beginning and/or end (with respect to time), depending on which was necessary, in order for the start and end times of all recorded signals to match. 

Specifically when dealing with patients with Apple Watch accelerometer recordings, additional data reduction was necessary; the accelerometer recordings tended to have much higher sampling rates compared to the ECG & watch heart rate recordings. For patients with recorded watch accelerometer data, moving average signal smoothing was performed individually on the Apple Watch accelerometer data (via 1 second moving average) prior to performing the 30 second moving averages on all DataFrames of the dataset. This functioned as an effective form of data reduction and smoothing of the curve. 

This entire pre-processing procedure was made into efficient and quick-to-use functions within a class, allowing other researchers on the team to efficiently perform all data reduction, cleaning, and beneficial preprocessing steps when dealing with this clinical data. 

### Feature Extraction 

With the pre-processed data, this already created the "first draft" DataFrame of the feature set to be used in the training of clustering algorithms, which included all recorded signals (ECG, watch heart rate, magnitude of watch acceleration, watch acceleration on x-axis, watch acceleration on y-axis, and watch acceleration on z-axis). However, the next step was to increase the size of this feature set through feature extraction. Adding onto the already created class and functions (described above), features were created dependent on already existing features. This included things such as ratios of already existing features, derivatives (involving Fourier Transforms), and many others. 

### Avoiding Multicollinearity and Standardization:

With the efficient click of a button, one can perform all preprocessing and feature selection for multiple signal recordings and multiple patients, all at once. After doing so, the next steps taken were checking for any strong correlation between features, which is done in order to avoid multicollinearity. If this problem is found, two choices can be made: either deleting one of the highly correlated features, or better yet, performing dimensionality reduction (with PCA) at a number of principal components that maintains as much variance as possible while simultaneously avoiding multicollinearity. In the example performed in the Jupyter notebook, one of the repetitive/unnecessary features was deleted, although PCA will be preferred in future modifications to this project. After such analysis and adjustments to the feature set, scaling was performed with StandardScaler(). At this point, the data is ready to be applied for the training of various clustering models.

### Creation and Training of Unsupervised Clustering Models:

Several unsupervised clustering models were attempted, including KMeans, Agglomerative Clustering (a type of hierarchical clustering), and Gaussian Mixture Models. For KMeans models, silhouette scores were used as a method of determining the optimal number of clusters to use, while agglomerative clustering in Python answers  that question for you with the visual display of a dendrogram. For Gaussian Mixture Models, Bayesian Information Criterion (BIC) and their gradients were mainly used as a method of determing the number of clusters, which is the preferred method over silhouette scores in this case, due to the more complex nature of clustering performed with GMM models. 

Lastly, several methods (adjusted random scores &amp; adjusted mutual information scores) were used in order to check for similarity among the created clustering models and to see which ones, if any, resulted in similar clusters. With these scores, it was proven that the results of clustering were consistently different among KMeans models & Gaussian Mixture Models (GMM). This makes reasonable sense due to the difference in clustering techniques taken by the two. GMM tends to use a more soft clustering technique, allowing overlaps in the clusters, while KMeans has more strict cutoffs/boundaries between clusters. GMM also has the ability to detect more complex patterns that can be hidden for KMeans models.

## Results:

The resulting clusters from all created models were analyzed over time to see if there were any noticeable patterns over time for certain clusters. Additionally, the noted seizure times were plotted along with the clusters, which allows visual analysis on whether the presence (or absence) of certain clusters can be associated with the occurence of a seizure or the preictal period of a seizure. For some patient (but not all), the absence of a cluster was associated with the occurence of a seizure. A KMeans model showed that a cluster associated with a low and constant heart rate was present before and after the period of a seizure, but it was no longer present in the time period of the seizure itself. This result aligns with research literature as well, explaining how the occurence of seizures tends to happen at an increased heart rate. This analysis of the different clusters/medical states (and their attributes) with respect to time will help as part of the larger goal in moving towards the ability to detect seizures ahead of time by the use of signal data collected from wearable devices, such as an Apple Watch. Further research into this project is necessary in order to find more substantial results; increased access to patient data in the future should greatly assist with this.

## Future Improvements:

As mentioned earlier, this research project can be improved down the line with access to more epilepsy patient data. Until then, there are still other aspects that could be improved in this project. In the future, the goal is to create one more compiled Pipeline including the class/functions created for data cleaning, data pre-processing, and feature extraction (as described above), followed by standardization/scaling, and ending with dimensionality reduction (to maintain variance while avoiding multicollinearity) via PCA with the number of components equaling "mle". This will automatically determine the number of components necessary for maintaining sufficient variance in the data. Another method for determining the number of components is to observe what number of components on an explained variance ratio plot maintains the desired variance (close to 100%). For a Pipeline, however, setting the number of components to "mle" is a more efficient pre-programmed method. This Pipeline would be even more simple than the already created class and functions, making it even easier for anyone to make further advancements in the research project. 

One of the next steps for advancements in this project is to experiment with the addition of new features, which can easily be done by simply adding on the necessary code to the already existing file for preprocessing and feature selection. Additionally, more filtering or experimentation with FFT (fast-fourier transform) results of the signals could also potentially help in further refining the feature set for improved outcomes in the resulting clusters from unsupervised learning. Finally, one of the final steps that would help in the analysis of the resulting clusters/states is to also plot the times in which patients take their medication, which will provide a visual representation as to how their medication affects their medical state. 



