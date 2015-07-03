import sys
import cv2
import numpy as np


def ann_gaze(img, gaze_info, face):
    landmark = face[u'landmark']
    for fea in landmark:
        if fea.find("eye") >= 0 or fea.find("nose") >= 0:
            point = landmark[fea]
            img = cv2.circle(img, (int(point["x"] * img.shape[1] / 100), int(point["y"] * img.shape[0] / 100)), 2,
                             (255, 0, 0), -1)
    circle_l = gaze_info[-2]
    circle_r = gaze_info[-1]
    cv2.circle(img, (circle_l[0], circle_l[1]), circle_l[2], (0, 255, 0), 1)
    cv2.circle(img, (circle_l[0], circle_l[1]), 2, (0, 0, 255), 1)
    cv2.circle(img, (circle_r[0], circle_r[1]), circle_r[2], (0, 255, 0), 1)
    cv2.circle(img, (circle_r[0], circle_r[1]), 2, (0, 0, 255), 1)


def get_eye_pos(landmark, width, height):
    # position of eyes, (left, right, up, down)
    pos_l = []
    pos_l.append(landmark["left_eye_left_corner"]["x"] * width / 100)
    pos_l.append(landmark["left_eye_right_corner"]["x"] * width / 100)
    pos_l.append(landmark["left_eye_top"]["y"] * height / 100)
    pos_l.append(landmark["left_eye_bottom"]["y"] * height / 100)
    pos_r = []
    pos_r.append(landmark["right_eye_left_corner"]["x"] * width / 100)
    pos_r.append(landmark["right_eye_right_corner"]["x"] * width / 100)
    pos_r.append(landmark["right_eye_top"]["y"] * height / 100)
    pos_r.append(landmark["right_eye_bottom"]["y"] * height / 100)
    return pos_l, pos_r


def get_eye_coords(landmark, width, height):
    coord_l = []
    coord_r = []
    for name in eye_profile:
        name_l = "left_" + name
        coord_l.append((landmark[name_l]["x"] * width / 100, landmark[name_l]["y"] * height / 100 ))
        name_r = "right_" + name
        coord_r.append((landmark[name_r]["x"] * width / 100, landmark[name_r]["y"] * height / 100 ))
    return coord_l, coord_r


def transform_coord(coords, ratio):
    x = coords[0][0]
    y = coords[6][1]
    coords_ = []
    for co in coords:
        coords_.append(((co[0] - x) / ratio, (co[1] - y) / ratio))
    return coords_


def get_grad_img(img, coords):
    # gen grad img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(np.square(gradx) + np.square(grady))
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grad_mag)
    grad_mag = grad_mag / maxVal * 255
    grad_mag = grad_mag.astype(np.uint8)
    # apply mask of eye
    mask = np.zeros(gray.shape, dtype=np.uint8)
    corners = np.array([coords], dtype=np.int32)
    cv2.fillPoly(mask, corners, 255)
    grad_mag = cv2.bitwise_and(grad_mag, mask)
    return grad_mag


def get_pupil(grad, ratio, coord):
    circles = cv2.HoughCircles(grad, cv2.HOUGH_GRADIENT, 1.5, 20, param1=35, param2=18, minRadius=10, maxRadius=20)
    circles = np.uint16(np.around(circles))
    circle = circles[0, :][0]
    circle[0] = circle[0] * ratio + coord[0][0]
    circle[1] = circle[1] * ratio + coord[6][1]
    circle[2] = circle[2] * ratio
    return circle

    """for i in circles[0,:]:
        if ind==1:
            cv2.circle(img_eye_l, (i[0], i[1]), i[2], (0,255,0), 1)
            cv2.circle(img_eye_l, (i[0], i[1]), 2, (0,0,255), 1)
        ind+=1"""


eye_profile = ["eye_left_corner", "eye_lower_left_quarter", "eye_bottom", "eye_lower_right_quarter",
               "eye_right_corner", "eye_upper_right_quarter", "eye_top", "eye_upper_left_quarter"]
ratio_disp = 1.0
face_angle_const = 100
eye_angle_hori_const = 57
eye_angle_vert_const = -725
eye_angle_vert_intercept = 0.0312
k = 0


def get_gaze(img, face):
    width = img.shape[1]
    height = img.shape[0]

    landmark = face["landmark"]
    # position of left and right eye (left, right, up, down)
    pos_l, pos_r = get_eye_pos(landmark, width, height)
    ratio_l = (pos_l[3] - pos_l[2]) / 25
    ratio_r = (pos_r[3] - pos_r[2]) / 25
    # profile coordinates (8 points)
    coord_l, coord_r = get_eye_coords(landmark, width, height)

    img_eye_l = img[pos_l[2]:pos_l[3], pos_l[0]:pos_l[1]]
    img_eye_r = img[pos_r[2]:pos_r[3], pos_r[0]:pos_r[1]]
    img_eye_l = cv2.resize(img_eye_l, None, fx=1 / ratio_l, fy=1 / ratio_l, interpolation=cv2.INTER_CUBIC)
    img_eye_r = cv2.resize(img_eye_r, None, fx=1 / ratio_r, fy=1 / ratio_r, interpolation=cv2.INTER_CUBIC)
    coord_l_ = transform_coord(coord_l, ratio_l)
    coord_r_ = transform_coord(coord_r, ratio_r)

    grad_l = get_grad_img(img_eye_l, coord_l_)
    grad_r = get_grad_img(img_eye_r, coord_r_)


    gray_l = cv2.cvtColor(img_eye_l,cv2.COLOR_BGR2GRAY);
    gray_r = cv2.cvtColor(img_eye_r,cv2.COLOR_BGR2GRAY);

    lysum = gray_l.sum(axis=0)
    lxsum = gray_l.sum(axis=1)

    rysum = gray_r.sum(axis=0)
    rxsum = gray_r.sum(axis=1)


    #lxsum = cv2.reduce(grad_l, 2, cv2.REDUCE_SUM)


    circle_l = get_pupil(grad_l, ratio_l, coord_l)  # [centerx, centery, radius]
    circle_r = get_pupil(grad_r, ratio_r, coord_r)

    img = cv2.resize(img, None, fx=ratio_disp, fy=ratio_disp, interpolation=cv2.INTER_CUBIC)
    width_ = width * ratio_disp
    height_ = height * ratio_disp
    for fea in landmark:
        if fea.find("eye") >= 0 or fea.find("contour") >= 0:
            point = landmark[fea]
            img = cv2.circle(img, (int(point["x"] * img.shape[1] / 100), int(point["y"] * img.shape[0] / 100)), 2,
                             (0, 0, 255), -1)

    # cv2.imshow(filename, img)
    # face angle
    face_l_x = (landmark["contour_left1"]["x"] + landmark["contour_left2"]["x"] + landmark["contour_left3"]["x"]) / 3
    face_r_x = (landmark["contour_right1"]["x"] + landmark["contour_right2"]["x"] + landmark["contour_right3"]["x"]) / 3
    face_mid_x = (face_l_x + face_r_x) / 2
    nose_x = landmark["nose_contour_lower_middle"]["x"]
    #print (nose_x - face_mid_x)/(face_r_x-face_l_x)*face_angle_const
    face_angle = (nose_x - face_mid_x) / (face_r_x - face_l_x) * face_angle_const
    face_width = (face_r_x - face_l_x) * width / 100

    # eye angle horizontal
    eye_center_l = landmark["left_eye_center"]["x"] * width_ / 100
    pupil_centerx_l = circle_l[0] * ratio_disp
    eye_center_r = landmark["right_eye_center"]["x"] * width_ / 100
    pupil_centerx_r = circle_r[0] * ratio_disp
    diffx_l = (pupil_centerx_l - eye_center_l) / (face_r_x - face_l_x)
    diffx_r = (pupil_centerx_r - eye_center_r) / (face_r_x - face_l_x)
    eye_angle_x = (diffx_l + diffx_r) * eye_angle_hori_const
    #tokens = filename.split("_")

    # eye angle vertical
    eye_horiy_l = (coord_l[0][1] + coord_l[4][1]) / 2
    eye_verty_l = (coord_l[2][1] + coord_l[6][1]) / 2
    eye_horiy_r = (coord_r[0][1] + coord_r[4][1]) / 2
    eye_verty_r = (coord_r[2][1] + coord_r[6][1]) / 2
    diffy_l = (eye_verty_l - eye_horiy_l) / face_width
    diffy_r = (eye_verty_r - eye_horiy_r) / face_width
    #print 'Diffy l',diffy_l
    #print 'Diffy r',diffy_r
    eye_angle_y = (diffy_l + diffy_r + eye_angle_vert_intercept) * eye_angle_vert_const

    tf = True
    """eye_bottom_l = landmark["left_eye_bottom"]["y"]*height_/100
    pupil_centery_l = circle_l[1]*ratio_disp
    eye_bottom_r = landmark["right_eye_bottom"]["y"]*height_/100
    pupil_centery_r = circle_r[1]*ratio_disp
    diffy_l = (eye_bottom_l-pupil_centery_l)/(face_r_x-face_l_x)
    diffy_r = (eye_bottom_r-pupil_centery_r)/(face_r_x-face_l_x)"""

    if eye_angle_x + face_angle > 15 or eye_angle_x + face_angle < -15 or eye_angle_x > 30 or eye_angle_x < -30:
        tf = False
    if eye_angle_y > 5 or eye_angle_y < -15:
        tf = False

    return tf, face_angle, eye_angle_x, eye_angle_y, lxsum, lysum, rxsum, rysum, circle_l, circle_r
    #print str(diffy_l)+" "+str(diffy_r)+" "+str(diffy_l+diffy_r)+" "+str(int(tokens[3][:-1]))
    #cv2.waitKey()
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    import os
    from fpp import FacePP
    from landmarks import face

    fpp = FacePP()
    i = 0
    of = open('sumgray.txt', 'w')
    for file in os.listdir("img"):
        if file.endswith(".jpg"):
            img = cv2.imread('img/' + file)
            img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            i += 1
            f = face[file]
            ret = get_gaze(img, f)
            #ann_gaze(img, ret, face)
            #cv2.imshow(file, img)
            #cv2.waitKey()
            of.write("%s\n" % repr(ret[4:8]))
            print ret
