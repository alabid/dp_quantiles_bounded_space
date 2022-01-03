README

OVERVIEW

This folder contains the code required to generate the experimental results for real world data sets reported in the paper "Bounded Space Differentially Private Quantiles". There are 3 folders in this directory:
	1. Data - Stores the Taxi and Gas sensor data sets downloaded from the UCI repository.
	2. Code - Contains the code used to generate sketches (full space and low space with different approximation factors), as well as to run the exponential mechanism.
	3. Sketches - A directory that stores the sketches.

HOW TO RUN
	
	1. Download the taxi.csv.zip file from the data folder hosted at "https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge%2C+ECML+PKDD+2015", and unzip and move the train.csv file to the data folder of this README's parent directory. Download the data.zip file from the data folder hosted at "https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures", unzip the dowloaded folder, move the 'ethylene_CO.txt' file to the data folder and change the extension from '.txt' to .csv'.
	2. New sketches for the Taxi and Gas sensor data sets can be computed by running 
	python3 rw.py
	python3 rwb.py
respectively. These sketches are saved in the sketches folder with similar file names as the pre-computed sketches. The approximation factors alpha for which the sketches are to be generated can be set via the "approxes" list variable (line 17 in both scripts). The sketch sizes are reported at the end of the computation. In the results reported in the paper we multiply the low space sketch sizes by a factor of 3 to account for the fact that in the GK sketch we must keep track of 3 numbers for each list element (an element of the data set, and two numbers capturing the confidence interval for its rank).
	3. To run the exponential mechanism and get the absolute error between the true non-private quantile and the private estimates for the Taxi and Gas sensor data sets one can run
	python3 rw2.py
	python3 rwb2.py
respectively. The number of trials to run can be set by the "num_trials" variable (line 39 in both scripts). At the end the mean absolute error as well as the 10th and 90th percentiles of all errors generated are output in tabular form. Note that in the paper we report and graph relative error - this is computed simply by dividing the absolute error by the standard deviation of the data sets.