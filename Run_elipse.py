import os
import datetime
import time
import cv2

from Algorithm import Util, EllipseModel, RansacEllipse
import traceback


def run_elipse(folder, filename, threshold, inlier, threshold_outlier_count,
               min_r=20, max_r=100, sampling_fraction=0.25):

    print("Going to process file:%s" % filename)
    start_time = time.time()
    file_noisy_circle = os.path.join(folder, filename)
    try:
        np_image = cv2.imread(file_noisy_circle, cv2.IMREAD_GRAYSCALE)
        lst_all_points = Util.create_points_from_numpyimage(np_image/255)
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
        best_model = helper.run(start_time)
        print("Algorithm-complete")
        if best_model is None:
            print("ERROR! Could not find a suitable model. Try altering ransac-threshold and min inliner count")
            return
        #
        print("--- %s seconds ---" % (time.time() - start_time))
        # Generate an output image with the model circle overlayed on top of original image
        #
        now = datetime.datetime.now()
        file_result = ("%s-%s.png" % (filename, now.strftime("%Y-%m-%d-%H-%M-%S")))
        np_image = cv2.imread(file_noisy_circle, cv2.IMREAD_COLOR)
        new_points = EllipseModel.generate_points_from_circle(best_model)
        np_superimposed = Util.superimpose_points_on_image(np_image, new_points, 100, 255, 100)
        cv2.imwrite(file_result, np_superimposed)
        print("Results saved to file:%s" % (file_result))
        print("------------------------------------------------------------")

    except Exception as e:
        tb = traceback.format_exc()
        print("Error:%s while doing Algorithm on the file: %s , stack=%s" % (str(e), filename, str(tb)))
        print("------------------------------------------------------------")
        pass


if __name__ == '__main__':
    run_elipse("01", "img0001.jpg", 1.5, 50, 100, min_r=40, max_r=100)

