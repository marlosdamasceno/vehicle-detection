import numpy as np
import cv2
from skimage.feature import hog
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Correct the distortion of the image
def correct_distortion(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


# Draw the boxes on the images
def draw_boxes(img, b_boxes, color=(0, 0, 255), thick=2):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding b_boxes
    for bbox in b_boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with b_boxes drawn
    return draw_img


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(x_start_stop=(500, 1300), y_start_stop=(380, 750),
                 xy_window=(128, 128), xy_overlap=(0.75, 0.75)):
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to compute color histogram features
def color_hist(img, bins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    r_hist = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
    g_hist = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
    b_hist = np.histogram(img[:, :, 2], bins=bins, range=bins_range)
    # Generating bin centers
    bin_edges = r_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((r_hist[0], g_hist[0], b_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return r_hist, g_hist, b_hist, bin_centers, hist_features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    image_spatial_resized = cv2.resize(img, size)
    features = image_spatial_resized.ravel()
    # Return the feature vector
    return image_spatial_resized, features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, visualise=False, feature_vec=True):
    if visualise:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=feature_vec)
        return hog_image, features
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


# Function that extract the features of an array of images, used in the training step
# It can accept different configurations, including add spatial features, histogram features and/or HOG features
def images_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 1),
                    orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial=False, hist=False, HOG=True):
    # Check the training time for the SVC
    t = time.time()
    # List of features
    list_features = []
    feature_image = None
    # Iterate through the list of images
    for file in imgs:
        # Create a list to append feature vectors to
        features = []
        # Read in each one by one, using cv2 because it does not mess up with the scale of the values, all readings are
        # in the range of 0 to 255, regardless the image type be jpg or png
        image = cv2.imread(file)
        # Apply color conversion because the cv2 reads the image in BGR space
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If spatial features are enable
        if spatial:
            # Apply bin_spatial() to get spatial color features
            a, spatial_features = bin_spatial(feature_image, spatial_size)
            features.append(spatial_features)
        # If histogram features are enable
        if hist:
            # Apply color_hist()
            a, b, c, d, hist_features = color_hist(feature_image, hist_bins, hist_range)
            features.append(hist_features)
        # If HOG features are enable
        if HOG:
            # Call get_hog_features()
            if hog_channel == 'ALL':  # For all channels
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(
                        get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block,
                                         False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:  # For a single one
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                                False, feature_vec=True)
            features.append(hog_features)
        list_features.append(np.concatenate(features))  # Append all features
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract the features of ' + str(len(imgs)) + ' images')
    # Return list of feature vectors
    return list_features


# Train all images to make a SVC
def train_car_classify(car_features, not_car_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))
    # Shuffle the data to get different results each time
    shuffle_X, shuffle_Y = shuffle(scaled_X, y)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(shuffle_X, shuffle_Y, test_size=0.2, random_state=rand_state)
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    # Training
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    print("Total of features: " + str(shuffle_X.shape[0]))
    print("Feature shape: " + str(shuffle_X.shape))
    return svc, X_scaler


# Get the feature for a single image
# It can accept different configurations, including add spatial features, histogram features and/or HOG features
def single_image_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 1),
                          orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial=False, hist=False,
                          HOG=True):
    # Create a list to append feature vectors to
    features = []
    feature_image = None
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    # If spatial features are enable
    if spatial:
        # Apply bin_spatial() to get spatial color features
        a, spatial_features = bin_spatial(feature_image, spatial_size)
        features.append(spatial_features)
    # If histogram features are enable
    if hist:
        # Apply color_hist() also with a color space option now
        a, b, c, d, hist_features = color_hist(feature_image, hist_bins, hist_range)
        features.append(hist_features)
    # If HOG features are enable
    if HOG:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block,
                                                     False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                            False, feature_vec=True)
        # Append all features together
        features.append(hog_features)
    # Return list of feature vectors
    return np.concatenate(features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 1), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel='ALL', spatial=False, hist=False, HOG=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_image_features(test_img, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial=spatial, hist=hist, HOG=HOG)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Get a blank heat map image
def get_zero_heat(img):
    return np.zeros_like(img).astype(np.float)


# Add the heat to the image
def add_heat(image_undistorted, b_box_list):
    # Zero heat image
    heat_map = np.zeros_like(image_undistorted[:, :, 0]).astype(np.float)
    # Iterate through list of bboxes
    for box in b_box_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heat_map  # Iterate through list of bboxes


# Apply the threshol to the heat image
def apply_heat_threshold(heat_map, threshold=3):
    # Zero out pixels below the threshold
    heat_map[heat_map <= threshold] = 0
    # Return thresholded map
    return np.array(np.clip(heat_map, 0, 255), dtype=np.uint8)


# Draw the final boxes on the founded cars
def draw_labeled_b_boxes(img, labels, area_threshold=4096):
    img_copy = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        box_area = (np.max(nonzerox) - np.min(nonzerox)) * (np.max(nonzeroy) - np.min(nonzeroy))
        if box_area > area_threshold:
            # Draw the box on the image
            cv2.rectangle(img_copy, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img_copy
