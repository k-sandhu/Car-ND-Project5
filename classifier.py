from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from skopt import gp_minimize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from sklearn.externals import joblib
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import pandas as pd
from utils import *

def get_data(sample_size):
    # Divide up into cars and notcars
    images = glob.glob('./data/train/**/*.png', recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    if sample_size > 0:
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    return cars, notcars

def get_features(cars, notcars, cspace, orient, pix_per_cell, cell_per_block, hog_channel):

    car_features = extract_features(cars, cspace=cspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=cspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)  # Create an array stack of feature vectors
    X_scaler = StandardScaler().fit(X)  # Fit a per-column scaler
    scaled_X = X_scaler.transform(X)  # Apply the scaler to X
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))  # Define the labels vector

    return X, scaled_X, y

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=4, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

def bopt(svc_base, X_train, y_train, cv, n_calls, n_random_starts, n_points):
    def svc_base_objective(params):
        C, tol= params

        svc_base.set_params(C=C, tol=tol)
        return -np.mean(cross_val_score(svc_base, X_train, y_train, cv=cv, n_jobs=-1, scoring="accuracy", verbose=2))

    svc_base.verbose = False
    svc_space = [(1,6),            # C
                    (0.0001, 0.001)]          # tol

    svc_opt = gp_minimize(svc_base_objective, svc_space,
                          n_calls=n_calls, n_random_starts=n_random_starts, acq_func='gp_hedge',
                          acq_optimizer='lbfgs', x0=None, y0=None, random_state=None,
                          verbose=True, callback=None, n_points=n_points,
                          n_restarts_optimizer=5, xi=0.01, kappa=1.96,
                          noise='gaussian', n_jobs=-1)
    #joblib.dump(svc_opt, "svc_opt.pkl")
    return svc_opt.x

if __name__ == "__main__":
    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 0
    max_count = None

    n_calls = 20
    n_random_starts = 10
    n_points = 1000
    cv=20

    colorspace_options = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
    hog_channel_options = [0, 1, 2, 'ALL']

    orient = 9
    pix_per_cell = 8
    cell_per_block = 4

    cars, notcars = get_data(sample_size)

    metrics_list = ['index', 'cspace', 'hog_channel', 'time_hog', 'f_vec_len',
                                    'time_train_base', 'time_predict_base', 'score_base',
                                    'time_train_base_f_sel', 'time_predict_base_f_sel', 'score_base_f_sel', 'feature_mask', 'num_features', 'bopt_param'
                                    'time_train_f_sel_bopt', 'time_predict_f_sel_bopt', 'score_f_sel_bopt',
                                    'time_train_cv', 'time_predict_cv', 'score_f_sel_bopt_cv',
                    'n_calls', 'n_random_starts', 'n_points', 'cv', 'orient', 'pix_per_cell', 'cell_per_block',
                    'bopt_param_nof', 'time_train_bopt_nof', 'time_predict_bopt_nof', 'score_bopt_nof']

    metrics_dict = {k:None for k in metrics_list}
    metrics = pd.DataFrame(columns=metrics_list)#, index=[0])

    count = 0


    for cspace in colorspace_options:
        hog_channel = 'ALL'
        #for hog_channel in hog_channel_options:
        if hog_channel is 'ALL':

            if max_count is not None and max_count == count:
                break

            metrics_dict['index'] = count+1
            metrics_dict['n_calls'] = n_calls
            metrics_dict['n_random_starts'] = n_random_starts
            metrics_dict['n_points'] = n_points
            metrics_dict['cv'] = cv

            metrics_dict['orient'] = orient
            metrics_dict['pix_per_cell'] = pix_per_cell
            metrics_dict['cell_per_block'] = cell_per_block

            metrics_dict['cspace'] = cspace
            metrics_dict['hog_channel'] = hog_channel

            t=time.time()
            X, scaled_X, y = get_features(cars, notcars, cspace, orient, pix_per_cell, cell_per_block, hog_channel)
            t2 = time.time()
            time_hog = round(t2 - t, 2)
            print(time_hog, 'Seconds to extract HOG features...')
            metrics_dict['time_hog'] = time_hog

            X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2) # Split up data into randomized training and test sets

            f_vec_len = len(X_train[0])
            print('Feature vector length:', f_vec_len)
            metrics_dict['f_vec_len'] = f_vec_len

            ######## Train and time base model ##################################
            t = time.time()
            svc = LinearSVC(verbose=True, max_iter=2000)
            svc.fit(X_train, y_train)
            t2 = time.time()
            time_train_base = round(t2 - t, 2)
            metrics_dict['time_train_base'] = time_train_base

            t = time.time()
            score = svc.score(X_test,y_test)
            t2 = time.time()
            time_predict_base = round(t2 - t, 2)
            metrics_dict['time_predict_base'] = time_predict_base

            metrics_dict['score_base'] = round(score, 6)
            print('Base Linear SVC score is:', score)

            ######## Train and time base after feature selection ##################################
            X_scaler = StandardScaler()
            svc = LinearSVC(verbose=2, max_iter=2000)
            sfm = SelectFromModel(svc)
            t = time.time()
            X_train_reduced = sfm.fit_transform(X_train, y_train)
            svc.fit(X_train_reduced, y_train)
            t2 = time.time()
            time_train_base_f_sel = round(t2 - t, 2)
            metrics_dict['time_train_base_f_sel'] = time_train_base_f_sel

            t = time.time()
            score = svc.score(sfm.transform(X_test), y_test)
            t2 = time.time()
            time_predict_base_f_sel = round(t2 - t, 2)
            metrics_dict['time_predict_base_f_sel'] = time_predict_base_f_sel

            feature_mask = sfm.get_support(indices=True)
            metrics_dict['score_base_f_sel'] = round(score, 6)
            metrics_dict['feature_mask_base'] = feature_mask

            num_features = X_train_reduced.shape[1]
            metrics_dict['num_features'] = num_features

            print('Number of features selected were: ', num_features)
            print('Base Linear SVC score after f selection is:', score)

            #BOpt without feature selection
            """t = time.time()
            svc.C, svc.tol = bopt(svc, X_train, y_train, cv, n_calls, n_random_starts, n_points)
            metrics_dict['bopt_param_nof'] = [svc.C, svc.tol]
            svc.fit(X_train, y_train)
            t2 = time.time()
            time_train_bopt = round(t2 - t, 2)
            metrics_dict['time_train_bopt_nof'] = time_train_bopt

            t = time.time()
            score = svc.score(X_test, y_test)
            t2 = time.time()
            time_predict_bopt_nof = round(t2 - t, 2)
            metrics_dict['time_predict_bopt_nof'] = time_predict_bopt_nof
            metrics_dict['score_bopt_nof'] = round(score, 6)
"""
            ######## Train and time bopt after feature selection ##################################
            t = time.time()
            svc.C, svc.tol = bopt(svc, X_train_reduced, y_train, cv, n_calls, n_random_starts, n_points)
            metrics_dict['bopt_param'] = [svc.C, svc.tol]
            svc.fit(X_train_reduced, y_train)
            t2 = time.time()
            time_train_bopt = round(t2 - t, 2)
            metrics_dict['time_train_bopt'] = time_train_bopt

            t = time.time()
            score = svc.score(sfm.transform(X_test), y_test)
            t2 = time.time()
            time_predict_f_sel_bopt = round(t2 - t, 2)
            metrics_dict['time_predict_f_sel_bopt'] = time_predict_f_sel_bopt
            metrics_dict['score_f_sel_bopt'] = round(score, 6)

            ################# cv after bopt ########################
            X_scaler = StandardScaler()
            svc = LinearSVC(verbose=True, max_iter=2000)
            sfm = SelectFromModel(svc)
            X_train_reduced = sfm.fit_transform(X_train, y_train)
            svc.C, svc.tol = bopt(svc, X_train_reduced, y_train, cv, n_calls, n_random_starts, n_points)
            t = time.time()

            estimators = [('StandardScaler', X_scaler), ('SelectFromModel', sfm), ('LinearSVC', svc)]
            pipe = Pipeline(estimators)

            params = {}
            grid = GridSearchCV(pipe, params, cv=cv, verbose=2, scoring='accuracy', refit=True, n_jobs=-1)
            grid.fit(X_train, y_train)
            t2 = time.time()
            time_train_cv = round(t2 - t, 2)
            metrics_dict['time_train_cv'] = time_train_cv

            t = time.time()
            score = grid.score(X_test, y_test)
            t2 = time.time()
            time_predict_cv = round(t2 - t, 2)
            metrics_dict['time_predict_cv'] = time_predict_cv

            metrics_dict['score_f_sel_bopt_cv'] = round(score, 6)
            metrics = metrics.append(metrics_dict, ignore_index=True)
            print(metrics)

            if count == 0:
                joblib.dump(grid, ("grid-{0}-{1}.pkl").format(score, count + 1))
            elif score > metrics['score_f_sel_bopt_cv'].max():
                previous = glob.glob('grid*.pkl')
                try:
                    os.remove(previous)
                except OSError:
                    pass
                joblib.dump(grid, ("grid-{0}-{1}.pkl").format(score, count+1))


            metrics.to_csv("metrics.csv", columns=metrics_dict.keys(), index=False)
            count+=1

    metrics.to_csv("metrics.csv", columns=metrics_dict.keys(), index=False)