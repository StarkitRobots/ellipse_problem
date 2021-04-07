from typing import List, Tuple
import math
import random
from Algorithm import EllipseModel
import time
import cv2
import numpy as np
from scipy.spatial import ConvexHull


class FiveOfPoints(object):
    def __init__(self, p1, p2, p3, p4, p5):
        self.P1 =  p1
        self.P2 =  p2
        self.P3 = p3
        self.P4 = p4
        self.P5 = p5

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


def get_angle(model: EllipseModel, p):
    si = p[0] * model.B
    co = p[1] * model.A
    return (math.atan2(si, co) + math.pi) * 180 / math.pi


def compute_outlier_measure(distance, radius):
    delta = abs(distance - radius)
    mx = max(distance, radius)
    ratio = delta / mx
    return ratio


def rot(p, phi):
    x = p[0] * math.cos(phi) + p[1] * math.sin(phi)
    y = - p[0] * math.sin(phi) + p[1] * math.cos(phi)
    return (x, y)


def change(p, model: EllipseModel):
    x = p[0] - model.X
    y = p[1] - model.Y
    return (x, y)


def conv(five):
    points = five.return_five_points()

    #points = [(p.X, p.Y) for p in points]

    points_hull = cv2.convexHull(np.array(points))
    if len(points_hull) < 5:
        return True
    return False


def co_angle(five):
    points = five.return_five_points()

    points = [(p.X, p.Y) for p in points]



    points_hull = cv2.convexHull(np.array(points))
    if len(points_hull) < 5:
        return True
    return False

class RansacEllipse(object):

    def __init__(self):
        self._all_points: List[Point] = list()
        self._points: List[Point] = list()
        self.threshold_error: float = float("nan")
        self.threshold_inlier_count = float("nan")
        self.threshold_outlier_count = float("nan")
        self.max_iterations = 100
        self.min_points_for_model = 20
        self.min_r = 20
        self.max_r = 100

    def add_points(self, points):
        self._all_points.extend(points)

    def validate_hyperparams(self):
        if math.isnan(self.threshold_error):
            raise Exception("The property 'threshold_error' has not been initialized")

        if math.isnan(self.threshold_inlier_count):
            raise Exception("The property 'threshold_inlier_count' has not been initialized")

        if math.isnan(self.threshold_outlier_count):
            raise Exception("The property 'threshold_inlier_count' has not been initialized")


    def run(self, start_time) -> EllipseModel:
        self.validate_hyperparams()
        count = 0
        fives = self.generate_five_from_points()
        start_time = time.time()
        time1 = 0
        time2 = 0
        time3 = 0
        # for el in fives:
        #     time1_start = time.time()
        #     points = el.return_five_points()
        #     time1 += (time.time() - time1_start)
        #     time2_start = time.time()
        #     points = [(p.X, p.Y) for p in points]
        #
        #     nonconvex = False
        #     points_hull = cv2.convexHull(np.array(points))
        #     if len(points_hull) < 5:
        #         count+=1
        # for i in range(len(points)):
        #     #tmp = points.copy()
        #     for_check = points[i]
        #     time3_start = time.time()
        #     res = cv2.pointPolygonTest(points_hull, for_check, False)
        #     time3 += (time.time() - time3_start)
        #     #points = tmp
        #     if res == 1:
        #         count += 1
        #         break
        # time2 += (time.time() - time2_start)

        print(time1, time2, time3)
        print(count, time.time() - start_time)

        five: FiveOfPoints
        lst_five_scores = list()


        progress_count = 0
        count_of_fives_with_poor_inliers = 0
        start_time = time.time()
        times = []
        global if1
        if1 = 0
        if2 = 0
        if3 = 0
        if4 = 0
        if5 = 0
        for five in fives:
            if len(cv2.convexHull(np.array(five.return_five_points()))) < 5:
                continue
            #if five_index % 100 == 0:
                #print("PROGRESS:Processing five %d of %d, shortlisted=%d  poor inliers=%d" % (
                 #   progress_count, len(random_five_indices), len(lst_five_scores), count_of_fives_with_poor_inliers))

            try:

                temp_ellipse = EllipseModel.GenerateModelFrom5Points(five.return_five_points(), self.min_r,
                                                                     self.max_r, if2, if3, if4, if5)

            except Exception as e:
                #print("Could not generate Ellipse model. Error=%s" % (str(e)))
                continue

            inliers, goodness_score, inliners_r, res_el, t = self.get_inliers(temp_ellipse,
                                                                              [five.P1, five.P2, five.P3, five.P4,
                                                                               five.P5])

            count_inliers = len(inliers)

            if (count_inliers < self.threshold_inlier_count) or (inliners_r > self.threshold_outlier_count) or (
                    inliners_r > count_inliers) or (res_el < 0.4):
                # print("Skipping because of poor inlier count=%d and this is less than threshold=%f)" % (count_inliers, self.threshold_inlier_count))
                count_of_fives_with_poor_inliers += 1
                continue
            result = (temp_ellipse, inliers, five, res_el)

            lst_five_scores.append(result)

        print(count)
        print("--- done check inliers %s seconds ---" % (time.time() - start_time))
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

    def generate_five_from_points(self, ):

        lst = list()
        points = self._all_points
        if len(points) < 200:
            n = 100
        else:
            n = 200
        print(int(len(points)))
        samp = int(len(points) - n)
        for c in range(0, 10):

            for j in range(0, samp, 1):
                p0, p1, p2, p3, p4 = random.choices(points[j:j + n:1], k=5)
                five = FiveOfPoints(p0, p1, p2, p3, p4)
                lst.append(five)

            # for j in range(len(points), n, -1):
            #     p0, p1, p2, p3, p4 = random.choices(points[j-n:j:1], k=5)
            #     five = FiveOfPoints(p0, p1, p2, p3, p4)
            #     lst.append(five)
        for c in range(0, 3):
            p0, p1, p2, p3, p4 = random.choices(points[samp - 1:samp - 1 + n:1], k=5)
            five = FiveOfPoints(p0, p1, p2, p3, p4)
            lst.append(five)
        print(len(lst))
        return lst
        pass

    def el_line(self, model: EllipseModel, p):

        x2 = p[0]
        y2 = p[1]

        a = model.A
        b = model.B
        k = y2 / x2
        A = b ** 2 + (k * a) ** 2
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

    def gran(self, x, gap, gran):
        x_min = x - gap
        x_max = x + gap
        if x < gap:
            x_min = 0
        if x + gap > gran:
            x_max = gran
        return x_min, x_max

    def get_inliers(self, model: EllipseModel, exclude_points):
        start_time = time.time()
        Phi = model.Phi
        X = model.X
        Y = model.Y
        all_points = self._all_points
        _all_points = self._points
        threshold = self.threshold_error

        shortlist_inliners = list()
        inliners_r = list()
        sum_goodness_measure = 0
        n = 50
        ang = 360 / n
        anglesEllipse = []
        for i in range(n):
            anglesEllipse.append(AngleEllipse(ang * i, ang * (i + 1)))

        X_a = int(X)
        Y_a = int(Y)
        """
        a, b = self.gran(X_a, 140, 540)
        c, d = self.gran(Y_a, 140, 720)
        for o in range(a, b):
            for j in range(c, d):
                if _all_points[o][j]>0.5:

                    y_cartesian = image_ht - o - 1
                    p = Point(j,y_cartesian)
                    if p in exclude_points:
                        continue
        """
        for p in all_points:
            if p in exclude_points:
                continue
                # pt = p

            xy = rot((X, Y), Phi)
            X = xy[0]
            Y = xy[1]
            pt = change(p, model)
            pt = rot(pt, model.Phi)
            p1, p2, = self.el_line(model, pt)

            distance_from_circumfrence = math.sqrt((pt[0] - p1) ** 2 + (pt[1] - p2) ** 2)
            distance = math.sqrt(p1 ** 2 + p2 ** 2)
            distance_from_center = math.sqrt((pt[0]) ** 2 + (pt[1]) ** 2)
            threshold_r = distance * 0.5
            dist2 = math.sqrt((-p1) ** 2 + (-p2) ** 2)
            if (distance_from_circumfrence > threshold) and (distance_from_circumfrence < threshold_r):
                inliners_r.append(p)
                continue
            if distance_from_circumfrence > threshold:
                continue
            for i in anglesEllipse:
                i.check_angle(get_angle(model, pt))

            distance_from_center = math.sqrt((pt[0] - X) ** 2 + (pt[1] - Y) ** 2)
            outlier_goodness_measure = compute_outlier_measure(distance_from_center, dist2)
            sum_goodness_measure += outlier_goodness_measure

            shortlist_inliners.append(p)

        avg_goodness = 1.0;
        res = 0
        for i in anglesEllipse:
            if i.return_status():
                res += ang
        res = res / 360
        if len(shortlist_inliners) != 0:
            avg_goodness = sum_goodness_measure / len(shortlist_inliners)
        return shortlist_inliners, avg_goodness, len(inliners_r), res, time.time() - start_time
