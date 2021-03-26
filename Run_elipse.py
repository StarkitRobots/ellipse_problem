import skimage.io
import os
import datetime
from skimage import util
import time
import cv2

from Algorithm import Util
from Algorithm import EllipseModel
from Algorithm import RansacEllipse
import traceback


def run_elipse(folder, filename, threshold, inlier, threshold_outlier_count,
               min_r=20, max_r=100, sampling_fraction=0.25):
    print("Going to process file:%s" % filename)
    start_time = time.time()
    file_noisy_circle = os.path.join(folder, filename)
    try:
        np_image = skimage.io.imread(file_noisy_circle, as_gray=True)
        #np_image = cv2.imread(file_noisy_circle, cv2.IMREAD_GRAYSCALE)
        np_image = util.invert(np_image)
        print(len(np_image))
        # Iterate over all cells of the NUMPY array and convert to array of Point classes
        #
        lst_all_points = Util.create_points_from_numpyimage(np_image)
        print(len(lst_all_points))
        #
        # begin Algorithm
        #
        helper = RansacEllipse()
        helper.threshold_error = threshold
        helper.min_r = min_r
        helper.max_r = max_r
        helper.threshold_outlier_count = threshold_outlier_count
        helper.threshold_inlier_count = inlier
        helper.add_points(lst_all_points)
        helper.sampling_fraction = sampling_fraction
        best_model = helper.run()
        print("Algorithm-complete")
        if best_model == None:
            print("ERROR! Could not find a suitable model. Try altering ransac-threshold and min inliner count")
            return
        #
        print("--- %s seconds ---" % (time.time() - start_time))
        # Generate an output image with the model circle overlayed on top of original image
        #
        now = datetime.datetime.now()
        filename_result = ("%s-%s.png" % (filename, now.strftime("%Y-%m-%d-%H-%M-%S")))
        file_result = filename_result
        # Load input image into array
        np_image_result = skimage.io.imread(file_noisy_circle, as_gray=True)
        new_points = EllipseModel.generate_points_from_circle(best_model)
        np_superimposed = Util.superimpose_points_on_image(np_image_result, new_points, 100, 255, 100)
        #new_points = EllipseModel.generate_points(lst_all_points)
        #np_superimposed = Util.superimpose_points_on_image(np_image_result, new_points, 100, 255, 100)
        skimage.io.imsave(file_result, np_superimposed)


        print("Results saved to file:%s" % (file_result))
        print("------------------------------------------------------------")

    except Exception as e:
        tb = traceback.format_exc()
        print("Error:%s while doing Algorithm on the file: %s , stack=%s" % (str(e), filename, str(tb)))
        print("------------------------------------------------------------")
        pass


if __name__ == '__main__':

    """for i in range(120):
        string = "img0"
        if i/10 ==0:
            string = string+"00"
        if i/10>=1:
            string = string+"0"
        string = string + str(i)+".jpg"
        #"img0069.jpg"""""
    run_elipse("01", "img0000.jpg", 1.0, 50, 50, min_r=40, max_r=100, sampling_fraction=1.00)

