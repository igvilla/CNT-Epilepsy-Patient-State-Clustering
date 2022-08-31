import tools
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, butter
import importlib
import sys
import os
sys.path.insert(1, os.path.join("./tools/"))

# reload library
importlib.reload(sys.modules['helpers'])
from helpers import *

from tools import get_iEEG_data
from get_iEEG_data import *

from tools.eeg_wearable import eeg_wearable
from tools.plot_eeg_wearable import plot_eeg_wearable
from tools.get_iEEG_data import get_iEEG_data

from tools.wearable_df_plot_noEEG_3dAcc import wearable_df_plot_noEEG_3dAcc

from IPython.display import display 

import pytz



####################


    ### ### ### ### ### ### ### ### ### ### ### ### 
    ### ### ### ### PIPELINE START ### ### ### ###
    ### ### ### ### ### ### ### ### ### ### ### ### 

class pipeline_preprocessing_and_featureSelection_v2:

    def __init__(self, list_subject_names, list_subject_files, str_window_LengthDisp):


        self.list_subject_names = list_subject_names
        self.list_subject_files = list_subject_files

        
            # string for desired window length (non-overlapping sliding windows for the training set of machine learning models)
            # window length = window displacement 
            # Note: with current format, the window length used must be the same among all patients; this same window length will be used on all patients
        self.str_window_LengthDisp = str_window_LengthDisp
            # '30s' = 30 second sliding window length & window displacement 





    ### ### ### ###
    ### FUNCTION 1: for saving the dataframes for all desired & available signals + adding timeStamps & reindexing the indices to integers ### 
    ### ### ### ### 

    def preprocessing_and_featureSelection(self):

        # using a prevbiously created class fo the loading in of data, using all relevant patient data 

        # use function from the above class to save the ECG, watch HR, and watch accelerometer data into their own dataframe variables

        list_dfs_allSubjects = wearable_df_plot_noEEG_3dAcc(self.list_subject_names, self.list_subject_files).save_wearable()

        self.list_dfs_allSubjects = list_dfs_allSubjects

        # from the above step, all signal dataframes for all patients are saved into a list together

        # # # # # # # # 
        # # # # # # # # 
        # # # # # # # # 

        ### ### ### ###
        ### FUNCTION 2 (to be used within function 1): for data reduction/smoothing used SPECIFICALLY for watch_Acc 
        #                                   + computing resample moving averages to create sliding windows of the data (for ML) ### 
        ### ### ### ### 

        def data_smoothing_and_moving_window_averages(kept_dfs, str_window_LengthDisp):

            # if watch_Acc data is included in the list of signal dataframes, first smooth out this signal & perform data reduction on it 
            # (watch_Acc has significantly higher sample rate, so data reduction/smoothing on it alone will help)
            updated_dfs = []
            for df in kept_dfs: 
                df_new = df
                # if the dataframe is a watch_Acc dataframe:
                if 'Watch_Acc_Magnitude' in df_new:
                    # should smooth the data & perform data reduction (watch_Acc has a very high sampling rate):
                        # 1 second moving averages calculated on the data for watch Accedlerometer, which typically have sample rates greater than 1 Hz
                    df_reduced = df_new.resample('1s', on = 'timeStamp', closed='right', origin='start', label='right').mean()
                    # add timeStamp column
                    df_reduced['timeStamp'] = df_reduced.index
                    # # # # 
                    # in case any timeStamps are still in offset, convert to US Eastern time so that all times match 
                    df_reduced['timeStamp'] = df_reduced['timeStamp'].tz_convert('US/Eastern')
                    # now reindex with integers (rather than haveing indices remain as time stamps):
                    df_reduced['Index'] = list(range(len(df_reduced)))
                    df_reduced = df_reduced.set_index('Index')
                    df_reduced.index.astype('int64')

                    df_new = df_reduced

                # replace the dataframe in the list of dataframes with this newer & smoothed out version:
                updated_dfs.append(df_new)

            
            # next, resample & calculate moving averages (on ALL signals) for the desired window length; 
            # this is to be used for non-overlapping moving windows for the training of ML models

            # first intialize the list to hold the updated dataframes
            mov_window_dfs = []

            for df in updated_dfs:
                # perform moving average at desired window length in units of time (str_window_LengthDisp)
                reduced_df = df.resample(str_window_LengthDisp, on='timeStamp', closed='right', origin='start', label='right').mean()
                # # # # 
                reduced_df.index = reduced_df.index.tz_convert('US/Eastern')        
                # # # # 
                mov_window_dfs.append(reduced_df)




            # lastly, merge all the dataframes into one, with shared starting & ending times for the recordings (using pd.merge_asof)
                # first, the moving averages results in all signals sharing the same displayed sampling rate
                # next, merging the dataframes into one with pd.merge_adsof allows the start and end times of all the signals to be the same
                # --> all the signals are now aligned in time for the entirety of the data in the dataframe
            all_signals = pd.merge_asof(mov_window_dfs[0], mov_window_dfs[1], on='timeStamp', direction='nearest')

            # merge any possible remaining dataframes (would be the case if the patient has all 3 signals: ECG, watch_HR, & watch_Acc):
                # this condition is only true in the case of a patient that has data for all 3 signals used so far: ECG, watch HR, and watch accelerometer
            if len(mov_window_dfs) > 2:
                for i in range(2, len(mov_window_dfs)):
                    all_signals = pd.merge_asof(all_signals, mov_window_dfs[i], on='timeStamp', direction='nearest')


            # search for NaN values among the rows (representing the time gaps present in data)
                # NaN values result whenever there was a gap in recording time (i.e. the time between samples exceeded the window length used formoving averages)
            # this will save all row numbers in which NaN values are found (in any column)
            nan_rows = all_signals[all_signals.isna().any(axis=1)].index.tolist()


            # make the values in all columns equal NaN for rows where NaN values were found, so that time gaps are ignored across all signals...
                # NaN values are present in time gaps larger than the set window length (at which moving averages were calculated)
                # in this case, if a time gap is found for one signal, this step ignores the data found in that time period for the other signals and replaces it 
                # with NaN values as well, working as a way of continuing to declare that time period as a time gap to be ignored
            all_signals.loc[nan_rows, all_signals.columns.values.tolist()] = np.nan

            ### now, reindex with integers ###
            all_signals['Index'] = list(range(len(all_signals)))
            all_signals = all_signals.set_index('Index')
            all_signals.index.astype('int64')

            # version of the dataframe with all NaN values DROPPED (important for using it for ML model training)
            all_signals_noNans = all_signals.dropna()
            ### now, reindex with integers ###
            all_signals_noNans['Index'] = list(range(len(all_signals_noNans)))
            all_signals_noNans = all_signals_noNans.set_index('Index')
            all_signals_noNans.index.astype('int64')


            ### this version of the dataframe of all signals is the start of the feature set
                ### other features will be added onto it later, such as derivatives, ratios, FFTs, filters, etc...
            start_of_feature_set = all_signals_noNans



            return start_of_feature_set
            # end of function 2 (within function 1)

        # # # # # # # # 
        # # # # # # # # 
        # # # # # # # # 

        

        final_featSet_allSubjects = []

        # going thru each patient individually...
        for idx in range(len(self.list_subject_names)):


            # going thru each individual dataframe/signal in the list of signals/dataframes for a specific patient
            for df in self.list_dfs_allSubjects[idx]:
                # adding timeStamp column to all dataframes (this was moved to be the indices after resampling (moving averages) in the previous steps)
                df['timeStamp'] = df.index
                # now reindex with integers: change the indices from timeStamps to integers from 0 to len(df)-1
                df['Index'] = list(range(len(df)))
                # reorder the index integers to be in order
                df = df.set_index('Index')
                df.index.astype('int64')



            ### ### ### ###
            ### at this point, determine what you would want the window length to be; this can be done by looking thru the largest time gaps of data & finding a reasonable window length 
            ### that is long enough in order to NOT skip over instances where there are, for example, 30 seconds between samples
            ### ### ### ###



            # applying the described function above (function 2):
                # smooth out watch_acc data, if it exists + perform moving averages at the desired window length  for all signals/dataframes
            feat_df = data_smoothing_and_moving_window_averages(self.list_dfs_allSubjects[idx], self.str_window_LengthDisp)


            ####################################
            ### ADDING ADDITIONAL FEATURES NOW...
            ####################################


            ### ### ### ###
            ### STEP 1: taking care of all HR data first (before accelerometer), if it exists
            ### ### ### ### 


            #############


            ### ### 
            ### RATIO OF ECG HR & WATCH HR ### 
            ### ### 
            if 'ECG_HR' in feat_df and 'Watch_HR' in feat_df:
                feat_df['Ratio_ECG_to_Watch_HR'] = feat_df['ECG_HR'] / feat_df['Watch_HR']


            ### ### 
            ### DERIVATIVE OF ECG HR ### 
            ### ### 
            if 'ECG_HR' in feat_df:

                L = int((feat_df['timeStamp'][len(feat_df)  -1] - feat_df['timeStamp'][0]).total_seconds())  # entire length of recorded data, represented in units of seconds... 
                n = len(feat_df) # number of samples in the data...
                dx = L/n
                x = np.arange(0, L, dx, dtype='complex_')   # time (sec) ; x-axis

                fft_ecg_vals = np.fft.fft(feat_df['ECG_HR'])

                kappa_ecg = (2*np.pi/L)*np.arange(-n/2, n/2)
                kappa_ecg = np.fft.fftshift(kappa_ecg)      # reorder fft frequencies

                d_fft_ecg_vals = kappa_ecg * fft_ecg_vals * (1j)

                df_ecg = np.real(np.fft.ifft(d_fft_ecg_vals))

                df_ecg_feat = df_ecg.real
                feat_df['Derivative_ECG_HR'] = df_ecg_feat

            ### ### 
            ### DERIVATIVE OF WATCH HR ### 
            ### ### 
            if 'Watch_HR' in feat_df:

                L = int((feat_df['timeStamp'][len(feat_df)  -1] - feat_df['timeStamp'][0]).total_seconds())  # entire length of recorded data, represented in units of seconds... 
                n = len(feat_df) # number of samples in the data...
                dx = L/n
                x = np.arange(0, L, dx, dtype='complex_')   # time (sec) ; x-axis

                fft_watchHR_vals = np.fft.fft(feat_df['Watch_HR'])

                kappa_watchHR = (2*np.pi/L)*np.arange(-n/2, n/2)
                kappa_watchHR = np.fft.fftshift(kappa_watchHR)      # reorder fft frequencies

                d_fft_watchHR_vals = kappa_watchHR * fft_watchHR_vals * (1j)

                df_watchHR = np.real(np.fft.ifft(d_fft_watchHR_vals))

                df_watchHR_feat = df_watchHR.real
                feat_df['Derivative_Watch_HR'] = df_watchHR_feat
        

            ### ### 
            ### RATIO OF HR DERIVATIVES: ECG TO WATCH HR ### 
            ### ### 
            if 'Derivative_ECG_HR' in feat_df and 'Derivative_Watch_HR' in feat_df:
                feat_df['Ratio_ecgDeriv_to_watchHRderiv'] = feat_df['Derivative_ECG_HR'] / feat_df['Derivative_Watch_HR']



            ### ### ### ###
            ### STEP 2: taking care of all accelerometer data as well now, if it exists...
            ### ### ### ### 

            ### ### 
            ### RATIO OF HR (ECG & Watch_HR) to Watch_Acc_Magnitude ### 
            ### ### 
            if 'Watch_Acc_Magnitude' in feat_df and 'ECG_HR' in feat_df:

                feat_df['Ratio_ECG_to_watchAccMag'] = feat_df['ECG_HR'] / feat_df['Watch_Acc_Magnitude']

            if 'Watch_Acc_Magnitude' in feat_df and 'Watch_HR_HR' in feat_df:

                feat_df['Ratio_watchHR_to_watchAccMag'] = feat_df['Watch_HR'] / feat_df['Watch_Acc_Magnitude']


            ### ### 
            ### RATIO OF EACH OF THE 3 ACCELERATION DIMENSIONS TO EACH OTHER ### 
            ### ### 
            if 'Watch_Acc_X' in feat_df and 'Watch_Acc_Y' in feat_df:

                feat_df['Ratio_AccX_to_AccY'] = feat_df['Watch_Acc_X'] / feat_df['Watch_Acc_Y']

            if 'Watch_Acc_X' in feat_df and 'Watch_Acc_Z' in feat_df:

                feat_df['Ratio_AccX_to_AccZ'] = feat_df['Watch_Acc_X'] / feat_df['Watch_Acc_Z']

            if 'Watch_Acc_Y' in feat_df and 'Watch_Acc_Z' in feat_df:

                feat_df['Ratio_AccY_to_AccZ'] = feat_df['Watch_Acc_Y'] / feat_df['Watch_Acc_Z']


            ### ### 
            ### DERIVATIVE OF WATCH_ACC_MAGNITUDE ### 
            ### ### 
            if 'Watch_Acc_Magnitude' in feat_df:

                L = int((feat_df['timeStamp'][len(feat_df) - 1] - feat_df['timeStamp'][0]).total_seconds())  # entire length of recorded data, represented in units of seconds... 
                n = len(feat_df) # number of samples in the data...
                dx = L/n
                x = np.arange(0, L, dx, dtype='complex_')   # time (sec) ; x-axis

                fft_watchAcc_vals = np.fft.fft(feat_df['Watch_Acc_Magnitude'])

                kappa_watchAcc = (2*np.pi/L)*np.arange(-n/2, n/2)
                kappa_watchAcc = np.fft.fftshift(kappa_watchAcc)      # reorder fft frequencies

                d_fft_watchcc_vals = kappa_watchAcc * fft_watchAcc_vals * (1j)

                df_watchAcc = np.real(np.fft.ifft(d_fft_watchcc_vals))

                df_watchAcc_feat = df_watchAcc.real
                feat_df['Derivative_Watch_Acc_Mag'] = df_watchAcc_feat


            ### ### 
            ### RATIO OF HR DERIVATIVES (both ECG & Watch_HR) TO WATCH_ACC_MAGNITUDE DERIVATIVE ### 
            ### ### 
            if 'Derivative_Watch_Acc_Mag' in feat_df and 'Derivative_ECG_HR' in feat_df:

                feat_df['Ratio_ecgDeriv_to_watchAccDeriv'] = feat_df['Derivative_ECG_HR'] / feat_df['Derivative_Watch_Acc_Mag']

            if 'Derivative_Watch_Acc_Mag' in feat_df and 'Derivative_Watch_HR' in feat_df:

                feat_df['Ratio_watchHRderiv_to_watchAccDeriv'] = feat_df['Derivative_Watch_HR'] / feat_df['Derivative_Watch_Acc_Mag']


            ### ### 
            ### DERIVATIVES OF ACCELERATION IN ALL 3 DIMENSIONS ### 
            ### ### 
            if 'Watch_Acc_X' in feat_df:

                L = int((feat_df['timeStamp'][len(feat_df) - 1] - feat_df['timeStamp'][0]).total_seconds())  # entire length of recorded data, represented in units of seconds... 
                n = len(feat_df) # number of samples in the data...
                dx = L/n
                x = np.arange(0, L, dx, dtype='complex_')   # time (sec) ; x-axis

                fft_AccX_vals = np.fft.fft(feat_df['Watch_Acc_X'])

                kappa_AccX = (2*np.pi/L)*np.arange(-n/2, n/2)
                kappa_AccX = np.fft.fftshift(kappa_AccX)      # reorder fft frequencies

                d_fft_AccX_vals = kappa_AccX * fft_AccX_vals * (1j)

                df_AccX = np.real(np.fft.ifft(d_fft_AccX_vals))

                df_AccX_feat = df_AccX.real
                feat_df['Derivative_Acc_X'] = df_AccX_feat


            if 'Watch_Acc_Y' in feat_df:

                L = int((feat_df['timeStamp'][len(feat_df) - 1] - feat_df['timeStamp'][0]).total_seconds())  # entire length of recorded data, represented in units of seconds... 
                n = len(feat_df) # number of samples in the data...
                dx = L/n
                x = np.arange(0, L, dx, dtype='complex_')   # time (sec) ; x-axis

                fft_AccY_vals = np.fft.fft(feat_df['Watch_Acc_Y'])

                kappa_AccY = (2*np.pi/L)*np.arange(-n/2, n/2)
                kappa_AccY = np.fft.fftshift(kappa_AccY)      # reorder fft frequencies

                d_fft_AccY_vals = kappa_AccY * fft_AccY_vals * (1j)

                df_AccY = np.real(np.fft.ifft(d_fft_AccY_vals))

                df_AccY_feat = df_AccY.real
                feat_df['Derivative_Acc_Y'] = df_AccY_feat


            if 'Watch_Acc_Z' in feat_df:

                L = int((feat_df['timeStamp'][len(feat_df) - 1] - feat_df['timeStamp'][0]).total_seconds())  # entire length of recorded data, represented in units of seconds... 
                n = len(feat_df) # number of samples in the data...
                dx = L/n
                x = np.arange(0, L, dx, dtype='complex_')   # time (sec) ; x-axis

                fft_AccZ_vals = np.fft.fft(feat_df['Watch_Acc_Z'])

                kappa_AccZ = (2*np.pi/L)*np.arange(-n/2, n/2)
                kappa_AccZ = np.fft.fftshift(kappa_AccZ)      # reorder fft frequencies

                d_fft_AccZ_vals = kappa_AccZ * fft_AccZ_vals * (1j)

                df_AccZ = np.real(np.fft.ifft(d_fft_AccZ_vals))

                df_AccZ_feat = df_AccZ.real
                feat_df['Derivative_Acc_Z'] = df_AccZ_feat


            
            ### ### 
            ### RATIO OF Watch_Acc X, Y, Z (3 dimension) DERIVATIVES TO EACH OTHER ### 
            ### ### 
            if 'Derivative_Acc_X' in feat_df and 'Derivative_Acc_Y' in feat_df:

                feat_df['Ratio_DerivAccX_to_DerivAccY'] = feat_df['Derivative_Acc_X'] / feat_df['Derivative_Acc_Y']

            if 'Derivative_Acc_X' in feat_df and 'Derivative_Acc_Z' in feat_df:

                feat_df['Ratio_DerivAccX_to_DerivAccZ'] = feat_df['Derivative_Acc_X'] / feat_df['Derivative_Acc_Z']

            if 'Derivative_Acc_Y' in feat_df and 'Derivative_Acc_Z' in feat_df:

                feat_df['Ratio_DerivAccY_to_DerivAccZ'] = feat_df['Derivative_Acc_Y'] / feat_df['Derivative_Acc_Z']


            ##################################################################################################################################################################
            ##################################################################################################################################################################

            ###             IF ANY ADDITIONAL FEATURES WANT TO BE CREATED AND ADDED TO THE FUNCTION FOR FEATURE SELECTION, THIS IS THE SPOT IN WHICH TO DO IT              ###

            ##################################################################################################################################################################
            ##################################################################################################################################################################


            # finally, as last step before adding it to the list:
                # get ride of the 'Index_x' and 'Index_y' columns that may accidentally getting made 
            if 'Index_x' in feat_df:
                feat_df.drop('Index_x', axis=1, inplace=True)
            if 'Index_y' in feat_df:
                feat_df.drop('Index_y', axis=1, inplace=True)

            # add the finalized dataframe for a specific subject to the liat of dataframes for all subjects...
            final_featSet_allSubjects.append(feat_df)


        return final_featSet_allSubjects


    ### ### ### ###
    ### at this point, one should check if there is high correlation between any features (which one should try to avoid; avoid multicollinearity)
    ### If so, maybe delete one of those features that are highly correlated 
    ### ... OR, possibly use PCA, although one would need to know how to trace back to the original features from the principal components, since that will be useful when 
    ###         evaluating the properties of the resulting centroids of clusters & what features they are associated with 
    ###         ex: being able to determine whether a certain cluster is associated with high or low Heart Rate, high or low accelerometry, etc...
    ### ### ### ###


    ### ### ### ###
    ### at this point, one should perform standardization with StandardScaler()
    ### ### ### ###


    ### ### ### ### ### ### ### ### ### ### ### ### 
    ### ### ### ### PIPELINE END ### ### ### ###
    ### ### ### ### ### ### ### ### ### ### ### ### 



        

    ### ### ### ###
    ### at this point, create & train different models...
    ### ### ### ###

