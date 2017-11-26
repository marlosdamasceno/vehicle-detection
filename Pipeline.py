import pickle
import DataToPrint
import Helper
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


class Pipeline:
    def __init__(self):
        # Camera distortion such as the project four
        self.mtx = None
        self.dist = None
        # This object data_to_be_printed will contain relevant information to the project, also it will be
        # used to print information for further consulting
        self.data_to_be_printed = DataToPrint.DataToPrint()
        # Array for non car images
        self.not_cars = []
        # Array for car images
        self.cars = []
        # List of heat map images from frames. It will be used to average the heat maps
        self.heat_map_list = []

    def get_pickle_data(self, directory="camera_cal/", file_name="wide_dist_pickle.p", with_trained_data=True):
        # Read in the saved camera matrix and distortion coefficients
        # These are the arrays you calculated using cv2.calibrateCamera()
        dist_pickle = pickle.load(open(directory + file_name, "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        # In case to use a pre trained SVM
        if with_trained_data:
            # Read the pre trained SVM and pre tuned parameters
            self.data_to_be_printed.load_pickle()

    # When it needs to train the SVM
    def training_car_classify(self):
        # Get the features of the car images. This method can take a while
        car_features = Helper.images_features(self.cars,
                                              color_space=self.data_to_be_printed.color_space,
                                              spatial_size=self.data_to_be_printed.spatial_size,
                                              hist_bins=self.data_to_be_printed.hist_bins,
                                              hist_range=self.data_to_be_printed.hist_range,
                                              orient=self.data_to_be_printed.orient,
                                              pix_per_cell=self.data_to_be_printed.pix_per_cell,
                                              cell_per_block=self.data_to_be_printed.cell_per_block,
                                              hog_channel=self.data_to_be_printed.hog_channel,
                                              spatial=self.data_to_be_printed.enable_spatial,
                                              hist=self.data_to_be_printed.enable_hist,
                                              HOG=self.data_to_be_printed.enable_hog)
        # Get the features of the non car images. This method can take a while
        not_car_features = Helper.images_features(self.not_cars,
                                                  color_space=self.data_to_be_printed.color_space,
                                                  spatial_size=self.data_to_be_printed.spatial_size,
                                                  hist_bins=self.data_to_be_printed.hist_bins,
                                                  hist_range=self.data_to_be_printed.hist_range,
                                                  orient=self.data_to_be_printed.orient,
                                                  pix_per_cell=self.data_to_be_printed.pix_per_cell,
                                                  cell_per_block=self.data_to_be_printed.cell_per_block,
                                                  hog_channel=self.data_to_be_printed.hog_channel,
                                                  spatial=self.data_to_be_printed.enable_spatial,
                                                  hist=self.data_to_be_printed.enable_hist,
                                                  HOG=self.data_to_be_printed.enable_hog)
        # Train the SVM or better saying the SVC (Support Vector Classifier). Keep the svc and the scaler
        self.data_to_be_printed.svc, self.data_to_be_printed.scaler = Helper.train_car_classify(car_features,
                                                                                                not_car_features)
        self.data_to_be_printed.save_pickle()

    # Pipeline it self
    def pipeline(self, image):
        # Correct distortion
        image_undistorted = Helper.correct_distortion(image, self.mtx, self.dist)
        # Get the slide_windows
        slide_windows = Helper.slide_window()  # Windows of 128 by 128 pixels
        slide_windows2 = Helper.slide_window(xy_window=(96, 96), y_start_stop=(380, 572))  # Windows of 96 by 96 pixels
        # Get the images of the boxes, in case of print them
        image_boxes = Helper.draw_boxes(image_undistorted, slide_windows)
        image_boxes = Helper.draw_boxes(image_boxes, slide_windows2, color=(255, 0, 0))
        # Sum the slide windows
        slide_windows += slide_windows2
        # Search for the windows where there is a car
        car_boxes = Helper.search_windows(image_undistorted, slide_windows,
                                          self.data_to_be_printed.svc,
                                          self.data_to_be_printed.scaler,
                                          color_space=self.data_to_be_printed.color_space,
                                          spatial_size=self.data_to_be_printed.spatial_size,
                                          hist_bins=self.data_to_be_printed.hist_bins,
                                          hist_range=self.data_to_be_printed.hist_range,
                                          orient=self.data_to_be_printed.orient,
                                          pix_per_cell=self.data_to_be_printed.pix_per_cell,
                                          cell_per_block=self.data_to_be_printed.cell_per_block,
                                          hog_channel=self.data_to_be_printed.hog_channel,
                                          spatial=self.data_to_be_printed.enable_spatial,
                                          hist=self.data_to_be_printed.enable_hist,
                                          HOG=self.data_to_be_printed.enable_hog)
        # Draw the boxes where it possible found a car
        car_image_boxes = Helper.draw_boxes(image_undistorted, car_boxes)
        # Get the boxes for the first frames of the video
        if len(self.heat_map_list) < self.data_to_be_printed.frames_average:
            # Get the positions liked to be a car
            heat_map = Helper.add_heat(image_undistorted, car_boxes)
            # Append the to the list
            self.heat_map_list.append(heat_map)
            # Apply a threshold
            heat_map_final = Helper.apply_heat_threshold(heat_map, self.data_to_be_printed.threshold_heat_map_single)
            # Get the labels
            labels = label(heat_map_final)
            # Draw the boxes for the cars founded
            car_found = Helper.draw_labeled_b_boxes(image_undistorted, labels,
                                                    self.data_to_be_printed.threshold_car_area)
        else:  # After that average them
            # Pop the first map of the list
            self.heat_map_list.pop(0)
            # Get the positions liked to be a car
            heat_map = Helper.add_heat(image_undistorted, car_boxes)
            # Append the to the list
            self.heat_map_list.append(heat_map)
            # Get a black heat map
            heat_map_total = Helper.get_zero_heat(heat_map)
            # Sum all the heats
            for heat_map_uni in self.heat_map_list:
                heat_map_total += heat_map_uni
            # Apply a threshold
            heat_map_final = Helper.apply_heat_threshold(heat_map_total,
                                                         self.data_to_be_printed.threshold_heat_map_average)
            # Get the labels
            labels = label(heat_map_final)
            # Draw the boxes for the cars founded
            car_found = Helper.draw_labeled_b_boxes(image_undistorted, labels,
                                                    self.data_to_be_printed.threshold_car_area)

        self.data_to_be_printed.b_boxes_image = image_boxes
        self.data_to_be_printed.image_car_boxes = car_image_boxes
        self.data_to_be_printed.image_heat_map = heat_map_final
        self.data_to_be_printed.image_car_found = car_found

        return car_found

    def print_info(self):
        print("Configuration")
        print("Color Space: " + self.data_to_be_printed.color_space)
        print("Is spatial enable? " + ("Yes" if self.data_to_be_printed.enable_spatial else "No"))
        if self.data_to_be_printed.enable_spatial:
            print("Spatial size: " + str(self.data_to_be_printed.spatial_size))
        print("Is histogram enable? " + ("Yes" if self.data_to_be_printed.enable_hist else "No"))
        if self.data_to_be_printed.enable_hist:
            print("Bins of histogram: " + str(self.data_to_be_printed.hist_bins))
            print("Range of histogram: " + str(self.data_to_be_printed.hist_range))
        print("Is HOG enable? " + ("Yes" if self.data_to_be_printed.enable_hog else "No"))
        if self.data_to_be_printed.enable_hog:
            print("HOG orientation: " + str(self.data_to_be_printed.orient))
            print("HOG pixels per cell: " + str(self.data_to_be_printed.pix_per_cell))
            print("HOG cells per block: " + str(self.data_to_be_printed.cell_per_block))
            print("HOG channel: " + str(self.data_to_be_printed.hog_channel))
        print("Threshold of car area: " + str(self.data_to_be_printed.threshold_car_area))
        print("Quantity of frames to average: " + str(self.data_to_be_printed.frames_average))
        print("Threshold single heat map: " + str(self.data_to_be_printed.threshold_heat_map_single))
        print("Threshold average heat map: " + str(self.data_to_be_printed.threshold_heat_map_average))
        # Plot the image of the boxes of the search windows
        plt.imshow(self.data_to_be_printed.b_boxes_image)
        plt.figure()
        # Plot the images that "found" a car on it
        plt.imshow(self.data_to_be_printed.image_car_boxes)
        plt.draw()
        # Plot the image of the bounding boxes of the cars and the heat map, side by side
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(self.data_to_be_printed.image_car_found)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(self.data_to_be_printed.image_heat_map, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.draw()
        plt.show()
