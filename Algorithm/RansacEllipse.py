from typing import List, Tuple
import math
import random
from Algorithm import Point
from Algorithm import EllipseModel


class FiveOfPoints(object):
    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, p5: Point):
        self.P1: Point = p1
        self.P2: Point = p2
        self.P3: Point = p3
        self.P4: Point = p4
        self.P5: Point = p5
        pass

    def return_five_points(self):
        return self.P1, self.P2, self.P3, self.P4, self.P5


class AngleEllipse(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.status = False

    def check_angle(self, angle):
        if self.start < angle < self.end:
            self.status = True

    def return_status(self):
        return self.status


class RansacEllipse(object):
    """Implements Ransac algorithm for Ellipse model"""

    def __init__(self):
        # all points in the population
        self._all_points: List[Point] = list()

        # A point will be considered an inlier of a model circle if it is
        # within this distance from circumfrence of the model
        self.threshold_error: float = float("nan")

        # Minimum count of inliers needed for a model to be shortlisted
        self.threshold_inlier_count = float("nan")

        self.threshold_outlier_count = float("nan")

        # The algorithm will run for these many iterations
        self.max_iterations = 100

        # These many points will be selected at random to build a model
        self.min_points_for_model = 20
        self.min_r = 20
        self.max_r = 100
        # Out of the total population of fives generated, the actual number selected is a fraction
        # specified by the following attribute
        self.sampling_fraction = 0.25
        pass

    #
    # Should be called once to set the full list of data points
    #
    def add_points(self, points: List[Point]):
        self._all_points.extend(points)
        pass

    def validate_hyperparams(self):
        if math.isnan(self.threshold_error) == True:
            raise Exception("The property 'threshold_error' has not been initialized")

        if math.isnan(self.threshold_inlier_count) == True:
            raise Exception("The property 'threshold_inlier_count' has not been initialized")

        if math.isnan(self.threshold_outlier_count) == True:
            raise Exception("The property 'threshold_inlier_count' has not been initialized")

        if math.isnan(self.sampling_fraction) == True:
            raise Exception("The property 'sampling_fraction' has not been initialized")
        if (self.sampling_fraction <= 0) or (self.sampling_fraction > 1.0):
            raise Exception("The property 'sampling_fraction' should be between 0 and 1")

    def run(self) -> EllipseModel:
        self.validate_hyperparams()

        #
        # generate fives of points - find some temporary model to hold this model
        #
        print("Generating fives")
        fives = self.generate_five_from_points()
        print("Generating fives complete. Count=%d" % (len(fives)))
        #
        # for ever triagram find circle model
        #   find the circle that passes through those points
        #   Determine model goodness score
        #
        tri: FiveOfPoints
        lst_five_scores = list()
        #
        all_five_indices = list(range(0, len(fives)))
        fraction = self.sampling_fraction
        random_count = int(len(all_five_indices) * fraction)
        random_five_indices = random.sample(all_five_indices, random_count)
        # for trig_index in range(0,len(fives)):
        progress_count = 0
        count_of_fives_with_poor_inliers = 0

        # scope for improvement of performance by multithreading
        # if you use a 200X200 image, with salt peper ration of 0.85 and sample fraction of 0.2 then you can generate ample load to test multi-threading
        #
        for trig_index in random_five_indices:
            progress_count += 1
            tri = fives[trig_index]
            if trig_index % 100 == 0:
                print("PROGRESS:Processing five %d of %d, shortlisted=%d  poor inliers=%d" % (
                    progress_count, len(random_five_indices), len(lst_five_scores), count_of_fives_with_poor_inliers))
            try:
                temp_circle = EllipseModel.GenerateModelFrom5Points(tri.return_five_points(), self.min_r,
                                                                    self.max_r)
            except Exception as e:
                # print("Could not generate Ellipse model. Error=%s" % (str(e)))
                continue

            inliers, goodness_score, inliners_r, res_el = self.get_inliers(temp_circle,
                                                                           [tri.P1, tri.P2, tri.P3, tri.P4, tri.P5])


            count_inliers = len(inliers)
            # print("res of ell!!!!", res_el)
            # if (len(inliners_r) > self.threshold_outlier_count):
            if (count_inliers < self.threshold_inlier_count) or (inliners_r > self.threshold_outlier_count) or (
                    inliners_r > count_inliers) or (res_el < 0.4):
                # print("Skipping because of poor inlier count=%d and this is less than threshold=%f)" % (count_inliers, self.threshold_inlier_count))
                count_of_fives_with_poor_inliers += 1
                continue
            result = (temp_circle, inliers, tri, res_el)

            lst_five_scores.append(result)
        #
        # Sort fives with lowest error
        #
        sorted_five_inliercount = sorted(lst_five_scores, key=lambda x: (x[3], len(x[1])), reverse=True)
        if len(sorted_five_inliercount) == 0:
            print("Finished building shortlist of fives. No fives found. Quitting")
            return
        print("Finished building shortlist of fives. Count=%d, Max inlier count=%d" % (
            len(sorted_five_inliercount), len(sorted_five_inliercount[0][1])))

        print("best model score", sorted_five_inliercount[0][3])
        best_model = sorted_five_inliercount[0][0]
        print(" x y a b c", best_model.X, best_model.Y, best_model.A, best_model.B, best_model.Phi)
        return best_model

    '''
    Iterates over all points and generates fives
    @param points:Collection of points on which to iterate
    '''

    # def GenerateTrigamFromPoints(self,points:List[Point])->List[fiveOfPoints]:
    def generate_five_from_points(self, ):
        # Scope for performance improvement.
        # No need to iterate over lists 3 times. We could simply create a 3d Numpy array ,
        # the indices of the points along each of the 3 axis
        # all 3d points in this Numpy array would be our desired fives (except for points on all the diagonals)
        lst = list()
        points = self._all_points
        if len(points) < 200:
            n = 100
        else:
            n = 200
        print(int(len(points) / 100))
        samp = int(len(points) - n)
        for c in range(0, 30):  # (10 - int(len(points)/100))):

            for j in range(0, samp, 1):
                p0, p1, p2, p3, p4 = random.choices(points[j:j + n:1], k=5)
                five = FiveOfPoints(p0, p1, p2, p3, p4)
                lst.append(five)
        for c in range(0, 3):
            p0, p1, p2, p3, p4 = random.choices(points[samp - 1:samp - 1 + n:1], k=5)
            five = FiveOfPoints(p0, p1, p2, p3, p4)
            lst.append(five)

        return lst
        pass

        #

    # Returns all points which are within the tolerance distance from the circumfrence of the specified circle
    # Points in the exclusion list will not be considered.
    #
    def el_line(self, model: EllipseModel, p: Point):

        x2 = p.X
        y2 = p.Y

        a = model.A
        b = model.B
        k = y2 / x2
        A = b ** 2 + (k * a) ** 2
        """
        k = (y2-y1)/(x2-x1)
        d = (x2*y1 - x1*y2)/(x2-x1)
        A = b**2 + (k*a)**2B = 2 * k * d * a ** 2 - 2 * x1 * b ** 2 - 2 * k * y1 * a ** 2
        C = (x1 * b) ** 2 + (d * a) ** 2 + (y1 * a) ** 2 - 2 * d * y1 * a ** 2 - (a * b) ** 2
        D = B ** 2 - 4 * A * C"""
        C = -(b * a) ** 2
        D = - 4 * A * C
        ansx1 = (math.sqrt(D)) / (2 * A)
        ansy1 = k * ansx1
        ansx2 = (- math.sqrt(D)) / (2 * A)
        ansy2 = k * ansx2
        otr1 = math.sqrt((ansx1 - x2) ** 2 + (ansy1 - y2) ** 2)
        otr2 = math.sqrt((ansx2 - x2) ** 2 + (ansy2 - y2) ** 2)
        if otr1 > otr2:
            return ansx2, ansy2
        else:
            return ansx1, ansy1

    def get_inliers(self, model: EllipseModel, exclude_points: List[Point]) -> Tuple[List[Point], float, List[Point]]:
        Phi = model.Phi
        A = model.A
        B = model.B
        X = model.X
        Y = model.Y
        all_points = self._all_points
        threshold = self.threshold_error
        pt: Point
        p: Point
        shortlist_inliners = list()
        inliners_r = list()
        sum_goodness_measure = 0
        n = 50
        ang = 360 / n
        anglesEllipse = []
        for i in range(n):
            anglesEllipse.append(AngleEllipse(ang * i, ang * (i + 1)))
        for p in all_points:
            if p in exclude_points:
                continue
            # pt = p
            xy = self.rot(Point(X, Y), Phi)
            X = xy.X
            Y = xy.Y
            pt = self.change(p, model)
            pt = self.rot(pt, model.Phi)
            p1, p2, = self.el_line(model, pt)

            distance_from_circumfrence = math.sqrt((pt.X - p1) ** 2 + (pt.Y - p2) ** 2)
            distance = math.sqrt(p1 ** 2 + p2 ** 2)
            distance_from_center = math.sqrt((pt.X) ** 2 + (pt.Y) ** 2)
            # print("dist", distance_from_circumfrence)
            # squared = ((p.X - X)) ** 2 + ((p.Y - Y) ) ** 2
            # distance_from_center = math.sqrt(squared)
            # distance_from_circumfrence = math.fabs(distance_from_center - ((A+B)/2))
            threshold_r = distance * 0.5
            dist2 = math.sqrt((-p1) ** 2 + (-p2) ** 2)
            if (distance_from_circumfrence > threshold) and (distance_from_circumfrence < threshold_r):
                inliners_r.append(p)
                continue
            if distance_from_circumfrence > threshold:
                continue
            for i in anglesEllipse:
                i.check_angle(self.get_angle(model, pt))

            distance_from_center = math.sqrt((pt.X - X) ** 2 + (pt.Y - Y) ** 2)

            # outlier_goodness_measure = self.compute_outlier_measure(distance_from_center, (A+B)/2)
            outlier_goodness_measure = self.compute_outlier_measure(distance_from_center, dist2)
            sum_goodness_measure += outlier_goodness_measure

            shortlist_inliners.append(p)
        avg_goodness = 1.0;
        res = 0
        for i in anglesEllipse:
            if i.return_status():
                res += ang
        res = res / 360
        if (len(shortlist_inliners) != 0):
            avg_goodness = sum_goodness_measure / len(shortlist_inliners)
        return (shortlist_inliners, avg_goodness, len(inliners_r), res)
        pass

    def get_angle(self, model: EllipseModel, p: Point):
        si = p.X * model.B
        co = p.Y * model.A
        return (math.atan2(si, co) + math.pi) * 180 / math.pi

    #
    # Gives us a relative idea of how far away the point is from the circumfrence given
    #   the distance of the point from the center
    #   the radius of the circle
    #   Points on the circumfrence have a value of 0 increasin to 1 as we move away from the circumfrence radially
    #
    def compute_outlier_measure(self, distance, radius):
        delta = abs(distance - radius)
        mx = max(distance, radius)
        ratio = delta / mx
        return ratio

    #
    # Use the gradience descent algorithm to find the circle that fits the givens points
    # use the modelhint as a starting circle
    #
    #
    # Returns the specified count of random selection of points from the full data set
    #
    def select_random_points(self, count: int):
        count_original = len(self._all_points)
        if (count >= count_original):
            message = "The count of random points:%d canot exceed length of original list:%d" % (count, count_original)
            raise Exception(message)
        lst = random.sample(population=self._all_points, k=count)
        return lst

    def rot(self, p: Point, phi):
        x = p.X * math.cos(phi) + p.Y * math.sin(phi)
        y = - p.X * math.sin(phi) + p.Y * math.cos(phi)
        return Point(x, y)

    def change(self, p: Point, model: EllipseModel):
        X = p.X - model.X
        Y = p.Y - model.Y
        return Point(X, Y)
