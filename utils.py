import shutil
import os
import cv2
import numpy as np
from skimage.feature import hog
from moviepy.editor import ImageSequenceClip

def get_boxes():
    """
    Returns sliding window coordinates.
    :return: List of the form [left_x_cord, top_y_cord, right_x_coord, bottom_y_coord]
    """
    boxes = []

    box_sizes = [256]
    left_x_cords = [x for x in range(0,1280,12)]
    top_y_cords =  [y for y in range(360,720,12)]

    for box_size in box_sizes:
        for x_cord in left_x_cords:
            for y_cord in top_y_cords:
                if box_size+x_cord < 1280 and box_size+y_cord < 720:
                    boxes.append([x_cord, y_cord, x_cord+box_size, y_cord+box_size])

    return boxes

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec,block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(image, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=4, hog_channel='ALL'):
    # Create a list to append feature vectors to
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
    # Return list of feature vectors
    return hog_features

def draw_boxes_info(image, current_data):
    """
    Draws square boxes on the image give a list of top_left corner
    coordinates and side lengths.
    :param image: Image on which box is to be drawn.
    :param current_data:
    :return: Image with boxes drawn on it.
    """

    font_position1 = (50, 600)
    font_position2 = (50, 650)
    font_scale = .4
    font_thickness = 1

    locations = current_data["locations"] #returns x1, y1, x2, y2
    frame_num = "Frame Number: " + str(current_data["frame_num"])

    for box in locations:
        box_text = ("Box locations are x1: {0}, y1: {1}, x2: {2}, y2: {3}").format(box[1],box[3],box[0],box[2])

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.putText(image, box_text, font_position1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness, cv2.LINE_AA)

    cv2.putText(image, frame_num, font_position2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA)

    return image

def draw_heat_map(image, heat_map):
    """
    Overlays heat map on the image.
    :param image: Three channel image.
    :param heat_map: Three channel heat map array.
    :return:
    """
    return image

def predict(model, features):
    """
    Takes in a image, makes sure it is of correct size and returns 1 if a
    car if found in the image. Returns zero otherwise.
    :param model: Trained model with a predict(features) method.
    :param features: Feature vector derived using the same pipeline that was
    used on training data.
    :return: 1 if car found in the image. 0 otherwise.
    """
    result = model.predict(features)
    return result


def write_video(project_video_output, output_folder, fps=20):
    """
    Reads images from the directory and outputs a video
    :param project_video_output: Name of output video
    :param output_folder: Location of images
    :param fps: Number of frames per second
    :return: None
    """
    print("Creating video {}, FPS={}".format(project_video_output, fps))
    clip = ImageSequenceClip(output_folder, fps)
    clip.write_videofile(project_video_output)

def clean_output_folder(output_folder):
    """
    Deletes the output images from the previous run
    :param output_folder: Folder where images are saved.
    :return: None
    """
    for root, dirs, files in os.walk(output_folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def get_overlap(box1, box2):
    """

    :param box1:
    :param box2:
    :return:
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 < x2 or y1 < y2:
        return None

    return [x1, y1, x2, y2]


