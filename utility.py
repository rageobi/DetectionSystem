import math
from multiprocessing import Pool
from cvlib.object_detection import draw_bbox
import cvlib as cv
import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from scipy.ndimage import find_objects
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud as rpc
from nuscenes.utils.data_classes import LidarPointCloud as lpc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import os.path as osp
import matplotlib.patches as patches
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from densityscan import Cluster, ClusterLists
import random
import pickle
import time
from shapely.geometry import Polygon

svc = None
net = None
output_layers = None
classes = None
c_slide = None

def load_svc():
    filename = "data/svc_hope.p"
    svc = pickle.load(open(filename, 'rb'))
    return svc

def calculation_of_radar_data(radar):
    x_points = radar.points[0]
    y_points = radar.points[1]
    z_points = radar.points[2]
    x_comp_velocity = radar.points[8]
    y_comp_velocity = radar.points[9]
    x_velocity = radar.points[6]
    y_velocity = radar.points[7]

    velocity_phi = np.rad2deg(np.arctan2(y_velocity, x_velocity))
    point_dist = np.sqrt(x_points ** 2 + y_points ** 2 + z_points ** 2)
    point_phi = np.rad2deg(np.arctan2(y_points, x_points))
    point_rad_velocity = np.sqrt(x_comp_velocity ** 2 + y_comp_velocity ** 2)

    return point_dist, point_phi, point_rad_velocity, velocity_phi


def custom_map_pointcloud_to_image(nusc,
                                   pointsensor_token: str,
                                   camera_token: str,
                                   min_dist: float = 1.0,
                                   render_intensity: bool = False,
                                   verbose=False):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    # rpc.abidefaults()
    rpc.disable_filters()
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        pc = lpc.from_file(pcl_path)
    else:
        pc = rpc.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.

    point_dist, point_phi, point_rad_velocity, velocity_phi = calculation_of_radar_data(
        pc)

    detections_radial_velocity_kmph = point_rad_velocity * 3.6
    point_cluster = appendtoclusterlist(velocity_phi, point_dist)
    cluster_list = point_cluster.cluster_cluster(2.5, 5,verbose)

    detections_radial_velocity_kmph = np.reshape(
        detections_radial_velocity_kmph, (1, detections_radial_velocity_kmph.shape[0]))
    d_phi = np.reshape(
        point_phi, (1, point_phi.shape[0]))
    d_dist = np.reshape(
        point_dist, (1, point_dist.shape[0]))
    velocity_phi = np.reshape(
        velocity_phi, (1, velocity_phi.shape[0]))
    points = np.append(pc.points, velocity_phi, axis=0)
    points = np.append(points, d_phi, axis=0)
    points = np.append(points, d_dist, axis=0)
    points = np.append(points, detections_radial_velocity_kmph, axis=0)
    #mask = np.where(points[18, :] >= -200)
    #pos = points[:, mask]
    #points = np.reshape(points, (points.shape[0], points.shape[2]))
    pc.points = points

    cs_record = nusc.get('calibrated_sensor',
                          pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        #assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar!'
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / \
            (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    else:
        # Retrieve the color from the depth.
        x_array = pc.points[0]
        y_array = pc.points[1]

        # calculate the distances of the detections to the radar-sensor

        coloring = cluster_list

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(
        cs_record['camera_intrinsic']), normalize=True)
    points = np.append(points, pc.points[3:22, :], axis=0)
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    if verbose:
        print('     Total number of points in frame',points.shape[1])
    return points, coloring, im


def appendtoclusterlist(x, y):
    cl = ClusterLists()
    for data in zip(x, y):
        cl.append(Cluster(data[0], data[1]))
    return cl


def get_boxes_yolo(frame, method, point, visualize=False,verbose=False):

    if method == 2:
        #frame_copy = np.copy(frame)
        if point[3] < 20:
            frame_size = 200
        else:
            frame_size =100
    
        x1 = int(round(point[0])) - (frame_size)
        y1 = int(round(point[1])) - (frame_size)
        x2 = int(round(point[0])) + (frame_size)
        y2 = int(round(point[1])) + (frame_size)

        frame = np.array(frame.crop((x1, y1, x2, y2)))
    else:
        frame = np.array(frame)
        x1 = y1 = 0
    bbox, label, confidence = get_yolo_detections(frame, (x1, y1))
    if visualize:
        for i, box in enumerate(bbox):
            frame_copy = np.copy(frame)
            a = (box[0][0] - x1)
            b = (box[0][1] - y1)
            c = (box[1][0] - x1)
            d = (box[1][1] - y1)

            cv2.rectangle(frame_copy, (a, b), (c, d), (0, 255, 0))
            cv2.putText(frame_copy, label[i], (box[0][0]-x1, box[0]
                                               [1]-y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            plt.imshow(frame_copy)
            plt.show()
    return bbox
    """ final_bbox = []
    filter_label = ['car', 'bus', 'truck']
    if len(label) > 0:
        for i, s in enumerate(label):
            for item in filter_label:
                if item in s and np.all(bbox[i] > [1, 0, 0, 0]):
                    frame_copy = np.copy(frame)
                    a = (bbox[i][0], bbox[i][1])
                    b = (bbox[i][2], bbox[i][3])
                    final_bbox.append((a,b))
                    cv2.rectangle(frame_copy, a, b,(0,255,0))
                    plt.imshow(frame_copy)
                    plt.show()
    t1 = time.time()
    print(t1-t0)
    return final_bbox, x1, y1, p """


def load_net(weights_location='data/YOLOv3/yolov3.weights', config_location='data/YOLOv3/yolov3.cfg', names_location='data/YOLOv3/yolov3_classes.txt'):
    net = cv2.dnn.readNet(weights_location, config_location)
    classes = []
    with open(names_location, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return net, output_layers, classes


def get_yolo_detections(frame, primary_origin=(0, 0)):
    # Reference from https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
    global net, output_layers, classes
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [2, 5, 7]:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    bbox = []
    label = []
    confidence = []
    for i in indexes:
        i = i[0]
        box = boxes[i]
        x = int(box[0]) + primary_origin[0]
        y = int(box[1]) + primary_origin[1]
        w = int(box[2])
        h = int(box[3])
        bbox.append(((x, y), ((x+w), (y+h))))
        label.append(str(classes[class_ids[i]]))
        confidence.append(confidences[i])

    return bbox, label, confidence


def get_boxes_svm(frame, point, visualize=False, verbose=False):

    if point[3] < 15:
        imsize_x = 500
        imsize_y = 500

    elif point[3] < 20:
        imsize_x = 250
        imsize_y = 250

    elif point[3] < 30:
        imsize_x = 200
        imsize_y = 200

    elif point[3] < 40:
        imsize_x = 150
        imsize_y = 150

    elif point[3] < 50:
        imsize_x = 120
        imsize_y = 120

    elif point[3] < 70:
        imsize_x = 50
        imsize_y = 50

    else:
        imsize_x = False
    if imsize_x:
        if point[3] > 14:
            window_size_1 = int(0.5 * (imsize_x))
            window_size_2 = int(0.3 * (imsize_x))
        else:
            window_size_1 = int(0.65 * (imsize_x))
            window_size_2 = int(0.45 * (imsize_x))
        x1 = int(round(point[0])) - ((imsize_x)//2)
        y1 = int(round(point[1]))-((imsize_y)//2)
        x2 = int(round(point[0]))+int((imsize_x)//1.5)
        y2 = int(round(point[1]))+((imsize_y)//3)
        frame = frame.crop((x1, y1, x2, y2))

        frame = np.array(frame)
        sliding_window_1 = get_window_slides(
            frame, window_size_1, overlap=0.05)
        sliding_window_2 = get_window_slides(
            frame, window_size_2, overlap=0.10)
        sliding_windows = sliding_window_1 + sliding_window_2

        vehicle_slides = predict_vehicles_slides(frame, sliding_windows)
        proba_frame, calculated_slides = get_calculated_box(
            frame.shape, vehicle_slides)
        frame_slides_complete = frame_slides_canvas(frame, sliding_windows)
        frame_slides_refined = frame_slides_canvas(frame, vehicle_slides)
        frame_slides_final = frame_slides_canvas(frame, calculated_slides)

        if visualize:
            f, axes = plt.subplots(1, 3, figsize=(20, 100))
            axes[0].set_title("All Sliding Windows")
            axes[0].imshow(frame_slides_complete)

            axes[1].set_title("Refined Sliding Windows")
            axes[1].imshow(frame_slides_refined)

            axes[2].set_title("Final Prediction")
            axes[2].imshow(frame_slides_final)

            plt.show()

        final_boxes = []
        for j, slide in enumerate(calculated_slides):
            if (slide != None and len(slide) > 0):
                a = x1 + slide[0][0]
                b = y1 + slide[0][1]
                c = x1 + slide[1][0]
                d = y1 + slide[1][1]
                final_boxes.append([(a, b), (c, d)])
        return final_boxes


def get_marked_frames(nusc, pointsensor_token, camera_token, method=2, visualize_frames=False, visualize_sub_frames=False,verbose=False):
    p, color, frame = custom_map_pointcloud_to_image(
        nusc, pointsensor_token, camera_token,verbose=verbose)
    filtered_col = p[[0, 1, 18, 19, 20, 21], :]
    color = np.array(color).reshape(1, color.shape[0])
    new_p = np.append(filtered_col, color, axis=0)
    # Get all unique cluster values
    un = np.unique(color, axis=1)
    averages = []

    def restrict_dupli_frames(average, averages):
        flag = 1

        for avg in averages:
            if abs(avg[0] - average[0]) < 51 and abs(avg[1] - average[1]) < 45:
                flag = 0
                return False
        return True

    for i, val in enumerate(un[0], 0):
        mask = np.logical_and(new_p[6, :] == val, new_p[5, :] > 7)
        filtered_points = new_p[:, mask]

        if filtered_points.shape[1] > 0:
            average = np.mean(filtered_points, axis=1)
            if len(averages) == 0:
                averages.append(
                    [average[0], average[1], average[3], average[4]])
            else:
                if restrict_dupli_frames([average[0], average[1]], averages):
                    averages.append(
                        [average[0], average[1], average[3], average[4]])

    boxes = []
    box = []
    if verbose:
            print('     Total number of point regions to be verified:', len(averages))
    if method == 1:
        for average in averages:
            boxes.append(get_boxes_svm(frame, average, visualize_sub_frames,verbose))
    elif method == 2:
        for average in averages:
            boxes.append(get_boxes_yolo(frame, method, average, visualize_sub_frames,verbose))
    else:
        boxes.append(get_boxes_yolo(frame, method, (0, 0, 0, 0), visualize_sub_frames,verbose))
    frame = np.array(frame)
    
    for i, bbox in enumerate(boxes):
        if (bbox != None and len(bbox) > 0):
            for j in range(len(bbox)):
                a = bbox[j][0][0]
                b = bbox[j][0][1]
                c = bbox[j][1][0]
                d = bbox[j][1][1]
                if (len(box) < 1):
                    box.append([[a, b], [c, b], [c, d], [a, d]])
                    #cv2.rectangle(img, (a,b),(c,d), color=(0, 0, 255), thickness=2)
                    # plt.imshow(img)
                    # plt.show()
                else:
                    if check_box_area([[a, b], [c, b], [c, d], [a, d]], box, frame):
                        box.append([[a, b], [c, b], [c, d], [a, d]])
                        #cv2.rectangle(img, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                    else:
                        global c_slide
                        b1_area, b2_area = get_box_area(
                            c_slide, [[a, b], [c, b], [c, d], [a, d]])
                        if b2_area > b1_area:
                            box.remove(c_slide)
                            box.append([[a, b], [c, b], [c, d], [a, d]])
                            #cv2.rectangle(img, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                            #cv2.rectangle(img, (c_box[0][0], c_box[0][1]), (c_box[2][0], c_box[2][1]), color=(255, 0, 0), thickness=2)
                        #cv2.rectangle (imm,(a,b),(c,d), color=(0, 255, 0), thickness=2)
                        # plt.imshow(imm)
                        # plt.show()
    if verbose:
        print('     Total number of vehicle regions predicted in frame:',len(box))
    for rect in box:
        cv2.rectangle(frame, (rect[0][0], rect[0][1]), (rect[2][0], rect[2][1]), color=(
            0, 255, 0), thickness=2)
    if visualize_frames:
        # plt.imshow(frame)
        # plt.show()
        if verbose:
            print('     Visualising points and predicted frames')
        points_in_image(p, np.array(averages)[:, :2], color, frame)

    return frame


def points_in_image(points, averages, colouring, frame):
    frame_copy = np.copy(frame)
    fig, ax = plt.subplots()
    sc = ax.scatter(points[0, ], points[1, ], c=colouring[0], s=8, alpha=0.5)
    averages = np.transpose(averages)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    t = sc.get_offsets()

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}\n Velocitty Phi ={},\n Phi = {}\n dist={},\n Rad vel ={},\n cluster ={}".format(" ".join(list(map(str, ind["ind"]))),
                                                                                                   " ".join(
                                                                                                       str([points[18, n] for n in ind["ind"]])),
                                                                                                   " ".join(str([points[19, n]
                                                                                                                 for n in ind["ind"]])),
                                                                                                   " ".join(str([points[20, n]
                                                                                                                 for n in ind["ind"]])),
                                                                                                   " ".join(str([points[21, n]
                                                                                                                 for n in ind["ind"]])),
                                                                                                   " ".join(str([colouring[0, n] for n in ind["ind"]])))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)
    sc2 = ax.scatter(averages[0, ], averages[1, ], s=14, alpha=0.9)
    plt.imshow(frame_copy)
    plt.show()


def get_box_area(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    return poly_1.area, poly_2.area


def check_box_area(box1, boxes, frame, visualize=False):
    for box2 in boxes:
        if not (box2 == box1):
            iou = calculate_intersection(box1, box2)
            a1, a2 = get_box_area(box1, box2)
            if iou < 0.80*a1 and iou < 0.80*a2:
                continue
            else:
                global c_slide
                if visualize:
                    if (a1 > a2):
                        c1 = (0, 255, 0)
                        c2 = (255, 0, 0)
                    else:
                        c1 = (255, 0, 0)
                        c2 = (0, 255, 0)
                        cv2.rectangle(
                            frame, (box2[0][0], box2[0][1]), (box2[2][0], box2[2][1]), color=c2, thickness=2)
                        cv2.rectangle(
                            frame, (box1[0][0], box1[0][1]), (box1[2][0], box1[2][1]), color=c1, thickness=2)
                        plt.imshow(frame)
                        plt.show()
                c_slide = box2
                return False
    return True


def calculate_intersection(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area  # / poly_1.union(poly_2).area
    return iou


def get_window_slides(frame, window_size, overlap):
    assert frame.shape[1] > window_size
    window_slides = []
    # print(frame.shape[0],frame.shape[1],window_size)
    n_x_windows = int(frame.shape[1]//(window_size*overlap))
    n_y_windows = int(frame.shape[0]//(window_size*overlap))
    # print(n_x_windows,n_y_windows)
    y_window_seed = 0
    for i in range(0, n_y_windows):
        if (y_window_seed+window_size) < frame.shape[0]:
            x_window_seed = 0
            for j in range(0, n_x_windows):
                if (x_window_seed + window_size) < frame.shape[1]:
                    # print((x_window_seed,y_window_seed),(x_window_seed+window_size,y_window_seed+window_size))
                    window_slides.append(
                        [(x_window_seed, y_window_seed), (x_window_seed+window_size, y_window_seed+window_size)])
                    x_window_seed = int(x_window_seed + (window_size*overlap))
                else:
                    break
            y_window_seed = int(y_window_seed + (window_size*overlap))
        else:
            break
    return window_slides


def frame_slides_canvas(frame, slide_windows):
    frame_copy = np.array(frame)
    for slide_window in slide_windows:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        cv2.rectangle(frame_copy, (slide_window[0][0], slide_window[0][1]), (
            slide_window[1][0], slide_window[1][1]), (color), 1)
    return frame_copy


def predict_vehicles_slides(frame, slide_windows):
    vehicle_slides = []
    global svc
    for slide_window in slide_windows:
        sub_frame = frame[slide_window[0][1]: slide_window[1]
                          [1], slide_window[0][0]: slide_window[1][0], :]
        sub_frame = cv2.cvtColor(sub_frame, cv2.COLOR_RGB2YUV)
        if(sub_frame.shape[1] == sub_frame.shape[0] and sub_frame.shape[1] != 0):
            sub_frame = cv2.resize(sub_frame, (64, 64))
            hog_feat = get_hog_features(sub_frame)
            # prediction=classifier1.predict(j)
            prediction = svc.predict(hog_feat)
            if prediction == 1:
                vehicle_slides.append(slide_window)
    return vehicle_slides


def get_hog_features(frame, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, feature_vector=True, multichannel=None):
    normalized_blocks = []
    if visualize:
        normalized_blocks, hog_image = hog(
            frame[:, :, :], orientations, pixels_per_cell, cells_per_block, visualize=visualize, feature_vector=feature_vector)
        return normalized_blocks, hog_image
    else:
        for channel in range(frame.shape[2]):
            normalized_blocks.append(hog(frame[:, :, channel], orientations, pixels_per_cell,
                                         cells_per_block, visualize=visualize, feature_vector=feature_vector))
        normalized_blocks = [np.ravel(normalized_blocks)]
        return normalized_blocks


def get_calculated_box(frame_size, slide_windows):
    proba_frame = np.zeros((frame_size[0], frame_size[1]))
    for slide_window in slide_windows:
        proba_frame[slide_window[0][1]:slide_window[1][1],
                    slide_window[0][0]:slide_window[1][0]] += 1
    # print(proba_frame)
    proba_frame[proba_frame <= (len(slide_windows)//2)] = 0

    proba_frame, n_vehicles = label(proba_frame)
    calculated_slides = []
    detected_slides = find_objects(proba_frame)
    
    for y_row, x_col in detected_slides:
        calculated_slides.append(
            ((x_col.start, y_row.start), (x_col.stop, y_row.stop)))
        #cv2.rectangle (imm,(x.start,y.start),(x.stop,y.stop), color=(0, 255, 0), thickness=1)
        # plt.imshow(imm)
        # plt.show()
    return proba_frame, calculated_slides


def save_video(frames, filename, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = str('data/videos/')+filename+str(random.randint(0, 1000))+'.avi'
    video = cv2.VideoWriter(filename, fourcc, fps, (size[1], size[0]))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.destroyAllWindows()
    video.release()
    print(('Saved Video {} successfully').format(filename))

def main(method=2, visualize_frames=False,visualize_sub_frames=False,verbose=False):
    location = '/home/rageobi/Downloads/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=location, verbose=True)
    pointsensor_channel = 'RADAR_FRONT'
    camera_channel = 'CAM_FRONT'
    frames = []
    global net, output_layers, classes,svc
    if method > 1:
        net, output_layers, classes = load_net()
        if verbose:
            print('Loaded YOLO Net')
        filename = 'YOLOv3_'
    else:
        svc = load_svc()
        if verbose:
            print('Loaded SVM predictor')
        filename = 'HOG_SVM_'

    for scene in nusc.scene:
        if verbose:
            print('Scene description: ',scene['description'])
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        check_token = first_sample_token
        i = 0
        while (check_token != '') and scene['name'] == 'scene-0061' and i >= 0:
            if verbose:
                print('     -------------------New-Scene----------------')
            sample_record = nusc.get('sample', check_token)
            pointsensor_token = sample_record['data'][pointsensor_channel]
            camera_token = sample_record['data'][camera_channel]
            frames.append(get_marked_frames(
                nusc, pointsensor_token, camera_token, method, visualize_frames, visualize_sub_frames,verbose))
            check_token = sample_record['next']
            i += 1

    save_video(frames, filename, 12, frames[0].shape[:2])

t0 = time.time()
main(1)
t1 = time.time()
t =t1-t0
print('Time for HOG is:',t)

t0 = time.time()
main(2)
t1 = time.time()
t =t1-t0
print('Time for Yolo point method is:',t)

t0 = time.time()
main(3)
t1 = time.time()
t =t1-t0
print('Time for Yolo point is:',t)