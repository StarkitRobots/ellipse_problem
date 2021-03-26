import numpy as np
from scipy.constants import pt
import math

from Algorithm.Point import Point
from typing import List


class EllipseModel(object):
    """Represents a Eellipse model using its center point and a b """

    def __init__(self, center_x: float, center_y: float, a: float, b: float, phi: float):
        self.X: float = center_x
        self.Y: float = center_y
        self.A: float = a
        self.B: float = b
        self.Phi: float = phi

    def __str__(self):
        display = ("X=%f Y=%f A=%f B=%f Phi=%f" % (self.X, self.Y, self.A, self.B, self.Phi))
        return display

    #
    # Determine the center and radius of a circle which passes through the 3 points
    #
    def general_equation(x, y):
        x = float(x)
        y = float(y)
        return [x ** 2, x * y, y ** 2, x, y]

    @classmethod
    def GenerateModelFrom5Points(cls, points, min_r, max_r):

        ans = np.ones([5])
        coef_curve = np.array([cls.general_equation(point.X, point.Y) for point in points])
        itog = np.linalg.solve(coef_curve, ans)

        A = itog[0]
        B = itog[1]
        C = itog[2]
        D = itog[3]
        E = itog[4]
        F = -1

        vsp = (B ** 2 - 4 * A * C)
        ty = A + C
        delt = (A * C - B ** 2 / 4) * F + B * E * D / 4 - C * D ** 2 / 4 - A * E ** 2 / 4
        sig = (A * C - B ** 2)

        if ((ty * delt) < 0) and sig > 0:
            x0 = (2 * C * D - B * E) / vsp
            y0 = (2 * A * E - B * D) / vsp
            if (x0 > -100) and (x0 < 820) and (y0 > -100) and (y0 < 640):
                ab1 = (A * E ** 2 + C * D ** 2 - B * D * E + vsp * F)
                ab2 = math.sqrt((A - C) ** 2 + B ** 2)
                a = -math.sqrt(2 * ab1 * ((A + C) + ab2)) / vsp

                b = -math.sqrt(2 * ab1 * ((A + C) - ab2)) / vsp
                if a > max_r or b > max_r or a < min_r or b < min_r:
                    raise ValueError('t too big or too small')
                # print("b = ", b)
                # print("a = ", a )
                if max(a, b) / min(a, b) > 1.5:
                    raise ValueError('t too big or too small')
                if (a > 10) and (a < 250) and (b > 10) and (b < 250):
                    if B == 0 and A < C:
                        phi = 0
                    elif B == 0 and A > C:
                        phi = math.pi / 2
                    else:
                        phi = math.atan((C - A - ab2) / B)  # +math.pi#* 180 / math.pi

                        # phi = math.atan((2*B/(A - C)))/2
                    # print('True')

                    ellipse = EllipseModel(x0, y0, a, b, phi)
                    return ellipse
                else:
                    raise ValueError('bad ellipse')
            else:
                raise ValueError('bad ellipse')
        else:
            raise ValueError('bad ellipse')

    @classmethod
    def generate_points_from_circle(cls, model, distance=1):
        angleStart = 0
        phi = model.Phi
        A = model.A
        B = model.B
        # phi = math.radians(model.Phi)
        print("-----phi----", phi)
        # phi = (math.pi / 2 + phi)

        # elif phi > math.pi/2:
        #    phi = (math.pi*2-phi + math.pi / 2 )

        # A = max(model.A, model.B)
        # B = min(model.A, model.B)
        print("A", model.A)
        print("B", model.B)
        angleEnd = 2 * math.pi
        circumfrence = 4 * (math.pi * A * B + (A - B) ** 2) / (A + B)
        num = int(circumfrence / distance)

        angles = np.linspace(angleStart, angleEnd, num)

        line_a = np.linspace(-model.A, model.A, num)
        lst_points: List[pt.Point] = list()
        # print(model.Phi," eto")
        for idx in range(0, len(line_a)):
            x = line_a[idx]
            y = 0
            xx = x * math.cos(phi) + y * math.sin(phi) + model.X
            yy = -x * math.sin(phi) + y * math.cos(phi) + model.Y
            pt_new = Point(xx, yy)
            lst_points.append(pt_new)
        for idx in range(0, len(angles)):
            theta = angles[idx]
            x = A * math.cos(theta)
            y = B * math.sin(theta)
            xx = x * math.cos(phi) + y * math.sin(phi) + model.X
            yy = -x * math.sin(phi) + y * math.cos(phi) + model.Y
            pt_new = Point(xx, yy)
            lst_points.append(pt_new)
        return lst_points

    @classmethod
    def generate_points(cls, listp):
        lst_points: List[pt.Point] = list()
        for idx in range(100, 200):
            lst_points.append(listp[idx])

        return lst_points

