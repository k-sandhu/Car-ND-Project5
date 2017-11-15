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

def train_model(svc_base, X_train, y_train):
    def svc_base_objective(params):
        C, tol, max_iter = params

        svc_base.set_params(C=C, tol=tol, max_iter=max_iter)
        return -np.mean(cross_val_score(svc_base, X_train, y_train, cv=25, n_jobs=-1, scoring="accuracy", verbose=1))

    svc_space = [(1,6),            # C
                    (0.0001, 0.001),           # tol
                    (1000, 5000)]           # max_iter

    svc_opt = gp_minimize(svc_base_objective, svc_space,
                          n_calls=8, n_random_starts=3, acq_func='gp_hedge',
                          acq_optimizer='lbfgs', x0=None, y0=None, random_state=None,
                          verbose=True, callback=None, n_points=100,
                          n_restarts_optimizer=5, xi=0.01, kappa=1.96,
                          noise='gaussian', n_jobs=-1)
    #joblib.dump(svc_opt, "svc_opt.pkl")
    return svc_opt.x

if __name__ == "__main__":
    # Divide up into cars and notcars
    images = glob.glob('./data/train/**/*.png',recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    colorspace = 'RGB' #['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 4
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"

    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    X_scaler = StandardScaler()
    svc = LinearSVC()
    sfm = SelectFromModel(svc)
    X_train_reduced = sfm.fit_transform(X_train, y_train)
    svc.C, svc.tol, svc.max_iter = train_model(svc, X_train, y_train)

    estimators = [('StandardScaler', X_scaler), ('SelectFromModel', sfm), ('LinearSVC', svc)]
    pipe = Pipeline(estimators)

    # Check the training time for the SVC
    t=time.time()
    params = {}
    grid = GridSearchCV(svc, params, cv=25, verbose=1,scoring='accuracy')
    grid.fit(X, y)
    t2 = time.time()

    #joblib.dump(grid, 'svc_grid.pkl')

    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(grid.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', grid.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')