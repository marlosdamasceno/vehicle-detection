import Pipeline
from moviepy.editor import VideoFileClip

# Pipeline class
pipeline = Pipeline.Pipeline()

# Uncomment this to train
# pipeline.get_pickle_data(with_trained_data=False)
# pipeline.not_cars = glob.glob('data_sets/non-vehicles/**/*.png')
# pipeline.cars = glob.glob('data_sets/vehicles/**/*.png')
# print("Car images: " + str(len(pipeline.cars)) + "\nNot car images: " + str(len(pipeline.not_cars)))
# pipeline.training_car_classify()

# Uncomment this to test an image
# pipeline.get_pickle_data()
# import cv2
# image = cv2.imread('test_images/test6.jpg')
# pipeline.pipeline(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# pipeline.print_info()

# Uncomment this to output a video
# Get information already saved, such as the trained linear SVM. That is important to decrease the time well testing
# new parameters
pipeline.get_pickle_data()
# Get the video and output it
output = 'project_video_out.mp4'
clip = VideoFileClip("project_video.mp4")
out_clip = clip.fl_image(pipeline.pipeline)
out_clip.write_videofile(output, audio=False)
pipeline.print_info()
