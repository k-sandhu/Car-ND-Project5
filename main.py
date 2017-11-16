from utils import *
#from preprocess import *
from sklearn.externals import joblib
from datetime import datetime
from cars import Cars
from skimage.io import imsave
from scipy import misc

def pipeline(image, cars, model, frame_num, heat_map='n'):
    """
    Take video frame as input. Implements sliding window search for cars, draws
    bounding boxes on the frame and returns the new frame.

    import glob
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    sample_images = glob.glob('./data/test_images/*.jpg',recursive=True)
    image = mpimg.imread(sample_images[0])
    cars = Cars()
    frame_num = 0
    pipeline(image, cars, model, 0, heat_map='y')

    :param image: Frame capture of the video
    :param cars: Cars Object to track location of cars from frame to frame
    :return: Frame with bounded boxes
    """

    boxes = get_boxes() # Generate box coordinates for sliding window search
    locations = [] # list to track locations of cars
    car_num = 0
    count = 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in reversed(boxes):
        section = image[box[1]:box[3],box[0]:box[2],:]
        section = misc.imresize(section, (64, 64, 3))
        features = extract_features(section)
        is_car = model.predict(features)

        if is_car == 1:
            locations.append(box)
            car_num += 1
        count += 1

        #if count > 15:
        #    break
    print("Cars: ", car_num, "No cars:",count-car_num)
    current_data = cars.add_locations(frame_num, locations)
    image = draw_boxes_info(image, current_data)

    if heat_map is 'y':
         heat_map = current_data["heat_map"]
         image_with_heatmap = draw_heat_map(image, heat_map)
         return image, heat_map, image_with_heatmap


    output_images_folder_heatmaps = "./data/videos/heatmaps/"
    clean_output_folder(output_images_folder_heatmaps)
    timestamp = str(
        datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3])
    imsave((output_images_folder_heatmaps + "frame-{0}-{1}.jpg").format(timestamp, frame_num),
           current_data["heat_map"])
    return image

if __name__ == "__main__":
    project_video_output = './data/videos/project_video_output.mp4'
    project_video = "./data/videos/project_video.mp4"
    output_images_folder = "./data/videos/output_images/"

    vidcap = cv2.VideoCapture(project_video)
    success, image = vidcap.read()
    frame_num = 1
    cars = Cars() # create the car object
    model = joblib.load('./models/grid-0.965-1.pkl') # load trained linear SVC from a .pkl file

    clean_output_folder(output_images_folder)  # deleted previous transformed images stored in the output folder

    while frame_num < 250:
        #print('Frame Number: ', frame_num)  # Print out frame number

        # Call to main pipeline
        image = pipeline(image, cars, model, frame_num)

        timestamp = str(
            datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3])  # save with timestamp and frame number as file name

        imsave((output_images_folder + "frame-{0}-{1}.jpg").format(timestamp, frame_num), image)  # save frame as a JPEG file
        frame_num += 1  # Track frame number

        success, image = vidcap.read()

    # Read all written images and save as a video
    write_video(project_video_output, output_images_folder)
