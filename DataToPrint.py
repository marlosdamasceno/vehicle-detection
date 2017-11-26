import pickle


# Contains the important data and parameters for the pipeline
class DataToPrint:
    def __init__(self):
        # Iimage Original
        self.original_image = None
        # Image undistorted
        self.undistorted_image = None
        # Image of the boxes on the search window
        self.b_boxes_image = None
        # Image of the boxes where a cer was found
        self.image_car_boxes = None
        # Image of the heat map
        self.image_heat_map = None
        # Image of boxes of the cars already founded
        self.image_car_found = None

        # Threshold for a single image
        self.threshold_heat_map_single = 3
        # Threshold for the average image
        self.threshold_heat_map_average = 11
        # Number of frames to average
        self.frames_average = 10
        # Minimal size of the area of a found car
        self.threshold_car_area = 7000

        # Color space for the images
        self.color_space = 'RGB'
        # HOG orientation
        self.orient = 9
        # HOG pixels por cell
        self.pix_per_cell = 8
        # HOG pixel por block
        self.cell_per_block = 2
        # HOG number of channels
        self.hog_channel = 'ALL'
        # HOG is enable
        self.enable_hog = True
        # Spatial feature size
        self.spatial_size = (32, 32)
        # Spatial is enable
        self.enable_spatial = True
        # Histogram number of beans
        self.hist_bins = 32
        # Histogram range
        self.hist_range = (0, 256)
        # Histogram is enable
        self.enable_hist = True

        # SVC trained
        self.svc = None
        # SVC scaler
        self.scaler = None

    # Save the important parameters
    def save_pickle(self):
        dist_pickle = {}
        dist_pickle["threshold_heat_map_single"] = self.threshold_heat_map_single
        dist_pickle["threshold_heat_map_average"] = self.threshold_heat_map_average
        dist_pickle["frames_average"] = self.frames_average
        dist_pickle["threshold_car_area"] = self.threshold_car_area
        dist_pickle["color_space"] = self.color_space
        dist_pickle["orient"] = self.orient
        dist_pickle["pix_per_cell"] = self.pix_per_cell
        dist_pickle["cell_per_block"] = self.cell_per_block
        dist_pickle["hog_channel"] = self.hog_channel
        dist_pickle["enable_hog"] = self.enable_hog
        dist_pickle["spatial_size"] = self.spatial_size
        dist_pickle["enable_spatial"] = self.enable_spatial
        dist_pickle["hist_bins"] = self.hist_bins
        dist_pickle["hist_range"] = self.hist_range
        dist_pickle["enable_hist"] = self.enable_hist
        dist_pickle["svc"] = self.svc
        dist_pickle["scaler"] = self.scaler
        pickle.dump(dist_pickle, open("data_dist_pickle.p", "wb"))

    # Load the important parameters
    def load_pickle(self):
        dist_pickle = pickle.load(open("data_dist_pickle.p", "rb"))
        self.threshold_heat_map_single = dist_pickle["threshold_heat_map_single"]
        self.threshold_heat_map_average = dist_pickle["threshold_heat_map_average"]
        self.frames_average = dist_pickle["frames_average"]
        self.threshold_car_area = dist_pickle["threshold_car_area"]
        self.color_space = dist_pickle["color_space"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.hog_channel = dist_pickle["hog_channel"]
        self.enable_hog = dist_pickle["enable_hog"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.enable_spatial = dist_pickle["enable_spatial"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.hist_range = dist_pickle["hist_range"]
        self.enable_hist = dist_pickle["enable_hist"]
        self.svc = dist_pickle["svc"]
        self.scaler = dist_pickle["scaler"]
