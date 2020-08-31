import argparse
import multiprocessing
from functools import partial
import math
from multiprocessing import Pool
#from cvlib.object_detection import draw_bbox
#import cvlib as cv
import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
from scipy.ndimage.measurements import label
from scipy.ndimage import find_objects
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud as rpc
from nuscenes.utils.data_classes import LidarPointCloud as lpc
from matplotlib import pyplot as plt
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

# golbal variables
svc = None
net = None
output_layers = None
classes = None
c_slide = None
xscaler = None


def calculation_of_radar_data(radar):
    """
        Function to extract features from radar pointcloud data

        Parameters
        ----------
        :param radar: Pointcloud data

        Returns
        ----------
        point_dist -> array: Distance magnitude of the point from sensor
        point_phi -> array : Azimuth of the point from sensor
        point_rad_velocity -> array : Compensated radial velocity of the point
        velocity_phi -> array : Azimuth of the radial velocity vectors
    """
    ## Get required features from radar pointcloud
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
                                   pointsensor_token,
                                   camera_token,
                                   verbose=False):
    # Inspired from the NuScenes Dev-Kit
    """
        Helper function to retrieve the image coordinate transformed point coordinates, clusters mappings for the points and the image frame.

        Parameters
        ----------
        :param nusc: Nuscenes object
        :param pointsensor_token: Point sensor token
        :param cam_token: Camera sensor token
        :param verbose: Boolean variable to display console logs

        Returns
        ----------
        points -> ndarray: Points data transformed to Image coordinates
        coloring -> list : Cluster associated for points
        im -> PIL Image : Image frame for the instance
    """
    # rpc.abidefaults()
    ## Disable all the radar filter settings
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
    ## Convert from meters/h to Km/h
    detections_radial_velocity_kmph = point_rad_velocity * 3.6

    ## Get Clusterlist object for velocity vectors azimuth and point distance
    point_cluster = appendtoclusterlist(velocity_phi, point_dist)

    ## Cluster all points which are within 2.5 radians of vel_phi and 5m distance as same cluster
    cluster_list = point_cluster.cluster_cluster(2.5, 5)

    detections_radial_velocity_kmph = np.reshape(
        detections_radial_velocity_kmph, (1, detections_radial_velocity_kmph.shape[0]))
    d_phi = np.reshape(
        point_phi, (1, point_phi.shape[0]))
    d_dist = np.reshape(
        point_dist, (1, point_dist.shape[0]))
    velocity_phi = np.reshape(
        velocity_phi, (1, velocity_phi.shape[0]))

    ## append calculated features to the radar pointcloud
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

    ## Let the coloring be based on clusters formed
    coloring = cluster_list

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(
        cs_record['camera_intrinsic']), normalize=True)

    ## rebuilding the pointcloud features
    points = np.append(points, pc.points[3:22, :], axis=0)
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    if verbose:
        print('     Total number of points in frame', points.shape[1])
    return points, coloring, im


def appendtoclusterlist(x, y):
    """
        Append points to the Clusterlist

        Parameters
        ----------
        :param x: X cordinate of the clusterlist
        :param y: Y cordinate of the clusterlist

        Returns
        --------
        cl -> ClusterLists : ClusterList of all the points provided

    """
    cl = ClusterLists()

    ## Forming the clustelist based on data provided
    for data in zip(x, y):
        cl.append(Cluster(data[0], data[1]))
    return cl


def get_boxes_yolo(frame, method, point, visualize=False, verbose=False):
    """
        Helper function to predict the vehicle box coordinates through YOLO net approach 

        Parameters
        ----------
        :param frame: Image frame which needs to be predicted
        :param method: int which specifies the classifier type. (2 for Modified YOLOv3 and 3 for Original YOLOv3)
        :param point: The point data of the current frame instance
        :param visualize: Boolean variable to check if user needs to visualize region frames which are proposed and marked
        :param verbose: Boolean variable to display console logs


        Returns
        --------
        bbox -> list : Vehicle detected box coordinates

    """
    if method == 2:  ## Modified YOLOv3
        #frame_copy = np.copy(frame)

        ## Empirically define the region or sub-frame size based on point distance value
        if point[3] < 15:
            frame_size = 450

        elif point[3] < 20:
            frame_size = 200

        else:
            frame_size = 100

        ## Crop regions to form a new frame
        x1 = int(round(point[0])) - (frame_size)
        y1 = int(round(point[1])) - (frame_size)
        x2 = int(round(point[0])) + (frame_size)
        y2 = int(round(point[1])) + (frame_size)

        frame = np.array(frame.crop((x1, y1, x2, y2)))
    else:
        ## Original YOLOv3
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
            # cv2.putText(frame_copy, label[i], (box[0][0]-x1, box[0]
            #                                   [1]-y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            plt.imshow(frame_copy)
            plt.show()
    return bbox


def load_svc():
    """
        Function to load trained model for the MODEL B approach

        Returns
        ----------
        svc : The SVC model
    """
    filename = "data/svc_hope.p"
    svc = pickle.load(open(filename, 'rb'))
    return svc


def load_svc_2():
    """
        Function to load trained model for the MODEL A approach

        Returns
        ----------
        svc : The SVC model
        xscaler : The fitted scaler value
    """
    filename = "data/svmhopeful.p"
    svc = pickle.load(open(filename, 'rb'))
    filename = "data/xscalerhopeful.p"
    xscaler = pickle.load(open(filename, 'rb'))
    return svc, xscaler


# Python/Project/data/YOLOv3/yolov3.cfg data/YOLOv3/yolov3.weights


def load_net(weights_location='data/YOLOv3/yolov3.weights', config_location='data/YOLOv3/yolov3.cfg', names_location='data/YOLOv3/yolov3_classes.txt'):
    """
        Helper function to load the YOLO network

        Parameters
        ----------
        :param weights_location: Network weights file location
        :param config_location: Network conifg file location
        :param names_location: Network classes file location

        Returns
        --------
        net -> dnn : Loaded Network
        output_layers -> list : Network layers
        classes -> list : Class names

    """
    ## Load the net based on weights and config provided
    net = cv2.dnn.readNet(weights_location, config_location)
    #net = cv2.dnn_DetectionModel(config_location, weights_location)
    classes = []

    ## Load all the classes
    with open(names_location, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    ## Define the output layers built based on loaded net
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return net, output_layers, classes


def get_yolo_detections(frame, primary_origin=(0, 0)):
    # Reference - https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
    """
        Function to predict boxes through the loaded YOLO network

        Parameters
        ----------
        :param frame: Image frame which needs to be predicted
        :param primary_origin: Tuple with starting coordinates of image. (0,0) for uncropped image. But if region of the frame is sent, pass the starting coordinates of the region wrt to orginal uncropped frame

        Returns
        --------
        bbox -> list : Predicted bounding boxes
        label -> list : Predicted box labels
        confidence -> list : Predicted boxes confidence scores

    """
    
    global net, output_layers, classes
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
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

            ## Take the detections whose confidence score is greater than 0.5 and classes of the boxes are [car,bus,truck]
            if confidence > 0.5 and class_id in [2, 5, 7]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    ## All the boxes with scores greater than 0.5 and Non-Max Sopression greater than 0.4 are defined as predicted detections
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


def get_boxes_svm(frame=None, visualize=False, verbose=False, method=1, point=None):
    # Inspired from https://github.com/JunshengFu/vehicle-detection/blob/master/svm_pipeline.py
    """
        Helper function to predict the vehicle box coordinates through SVM classifier approach 

        Parameters
        ----------
        :param frame: Image frame which needs to be predicted
        :param visualize: Boolean variable to check if user needs to visualize region frames which are proposed and marked
        :param verbose: Boolean variable to display console logs
        :param method: int which specifies the classifier type. (1 for MODEL B and 0 for MODEL A)
        :param point: The point data of the current frame instance

        Returns
        --------
        final_boxes -> list : Vehicle detected box coordinates

    """
    ## Empirically define the region or sub-frame size based on point distance value
    if point[3] < 15:
        frame_size = 500
        #frame_size_y = 500

    elif point[3] < 20:
        frame_size = 250
        #frame_size_y = 250

    elif point[3] < 30:
        frame_size = 200
        #frame_size_y = 200

    elif point[3] < 40:
        frame_size = 150
        #frame_size_y = 150

    elif point[3] < 50:
        frame_size = 120
        #frame_size_y = 120

    elif point[3] < 70:
        frame_size = 50
        #frame_size_y = 50

    else:
        frame_size = False

    ## Empirically calculate the window sizes based on the frame size
    if frame_size:
        if point[3] > 14:
            window_size_1 = int(0.5 * (frame_size))
            window_size_2 = int(0.3 * (frame_size))
        else:
            window_size_1 = int(0.65 * (frame_size))
            window_size_2 = int(0.45 * (frame_size))
        
        ## Crop regions to form a new frame
        x1 = int(round(point[0])) - ((frame_size) // 2)
        y1 = int(round(point[1])) - ((frame_size) // 2)
        x2 = int(round(point[0])) + ((frame_size) // 1.5)
        y2 = int(round(point[1])) + ((frame_size) // 3)
        frame = frame.crop((x1, y1, x2, y2))

        ## Define the overlap value based on the SVM model/method
        if method == 0:
            overlap = 0.09
        else:
            overlap = 0.10
        frame = np.array(frame)

        ## Get all the windows
        sliding_window_1 = get_window_slides(
            frame, window_size_1, overlap=overlap)
        # sliding_window_1 = get_window_slides(
        # frame, window_size_2, overlap=0.10)
        sliding_windows = sliding_window_1  # + sliding_window_2

        ## Get all windows predicted as vehicles
        vehicle_slides = predict_vehicles_slides_2(
            frame, method, sliding_windows)
        #vehicle_slides = predict_vehicles_slides(frame, sliding_windows)

        ## Get the final bounding boxes based on vehicle window predictions
        proba_frame, calculated_slides = get_calculated_box(
            frame.shape, vehicle_slides)

        ## Draw all the windows/boxes on the image frame
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
            ## Convert the bounding boxes from sub-frame to image co-ordinates
            if (slide != None and len(slide) > 0):
                a = x1 + slide[0][0]
                b = y1 + slide[0][1]
                c = x1 + slide[1][0]
                d = y1 + slide[1][1]
                final_boxes.append([(a, b), (c, d)])
        return final_boxes


def get_marked_frames(nusc, pointsensor_token, camera_token, method=(2, 0), visualize_frames=False, visualize_sub_frames=False, verbose=False):
    """
        Main helper function which handles the calls to other helper function. Gets all the vehicle predicition boxes and the box marked frames. 

        Parameters
        ----------
        :param nusc: Nuscenes object
        :param pointsensor_token: Radar sensor token
        :param cam_token: Camera sensor token
        :param method: Tuple which specifies the (classifier,isParallel)
        :param validate_results: Boolean variable to check if user needs to validate results
        :param visualize_frames: Boolean variable to check if user needs to visualize fully marked image frames
        :param visualize_sub_frames: Boolean variable to check if user needs to visualize region frames which are proposed and marked
        :param verbose: Boolean variable to display console logs

        Returns
        --------
        frame -> ndarray : Marked image frames
        box -> list : Vehicle detected box coordinates

    """
    p, color, frame = custom_map_pointcloud_to_image(
        nusc, pointsensor_token, camera_token, verbose)

    ## Get only the X, Y and the calculated Radar features from the pointcloud
    filtered_col = p[[0, 1, 18, 19, 20, 21], :]

    ## Cluster information
    color = np.array(color).reshape(1, color.shape[0])

    ## Append both to a np array
    new_p = np.append(filtered_col, color, axis=0)
    ## Get all unique cluster values
    un = np.unique(color, axis=1)
    averages = []

    def restrict_dupli_frames(average, averages):
        """
        Checks if the "average" region is redundant for other "averages" regions
        """
        flag = 1

        for avg in averages:
            if abs(avg[0] - average[0]) < 51 and abs(avg[1] - average[1]) < 45:
                flag = 0
                return False
        return True

    ## Loop through unique cluster values
    for i, val in enumerate(un[0], 0):

        ## Getting all the filtered pointcloud data for a specific cluster value and also has compensated radial velocity above a threshold
        mask = np.logical_and(new_p[6, :] == val, new_p[5, :] > 7)
        filtered_points = new_p[:, mask]

        if filtered_points.shape[1] > 0:

            ## Average all the point cloud data and store it in a var
            average = np.mean(filtered_points, axis=1)
            if len(averages) == 0:
                averages.append(
                    [average[0], average[1], average[3], average[4]])
            else:
                ## Check for dupilcate frames and append only if not
                if restrict_dupli_frames([average[0], average[1]], averages):
                    averages.append(
                        [average[0], average[1], average[3], average[4]])

    boxes = []
    box = []
    if verbose:
        print('     Total number of point regions to be verified:', len(averages))

    ## method[0]= 0 = MODEL A,
    ##            1 = MODEL B,
    ##            2 = Modified YOLOv3,
    ##            3 = Original YOLOv3

    ## method[1]= 0 = No parallel processing,
    ##            1 = Parallel processing,
    if method[0] <= 1:
        if (method[1]) == 1:
            ## Open process pool and get bounding boxes through get_boxes_svm(...) for every "average" radar point
            pool = multiprocessing.Pool()
            func = partial(get_boxes_svm, frame,
                           visualize_sub_frames, verbose, method[0])
            boxes = (pool.map(func, averages))
            pool.close()
            pool.join()
        else:
            ## Get bounding boxes through get_boxes_svm(...) for every "average" radar point
            for average in averages:
                boxes.append(get_boxes_svm(
                    frame, visualize_sub_frames, verbose, method[0], average))
    elif method[0] == 2:

        ## Get bounding boxes through get_boxes_yolo(...) for every "average" radar point
        for average in averages:
            boxes.append(get_boxes_yolo(
                frame, method[0], average, visualize_sub_frames, verbose))
    else:
        ## Get bounding boxes through get_boxes_yolo(...) for every "average" radar point
        boxes.append(get_boxes_yolo(
            frame, method[0], (0, 0, 0, 0), visualize_sub_frames, verbose))
    frame = np.array(frame)

    for i, bbox in enumerate(boxes):
        if (bbox != None and len(bbox) > 0):
            for j in range(len(bbox)):
                if (bbox[j] != None and len(bbox[j]) > 0):
                    a = bbox[j][0][0]
                    b = bbox[j][0][1]
                    c = bbox[j][1][0]
                    d = bbox[j][1][1]
                    if (len(box) < 1):
                        box.append([[a, b], [c, b], [c, d], [a, d]])
                        #cv2.rectangle(frame, (a,b),(c,d), color=(0, 0, 255), thickness=2)
                        # plt.imshow(frame)
                        # plt.show()
                    else:

                        ## Check if an approximate bounding box is predicted already and if it's predicted already add/retain the one with more area and remove the other
                        if check_box_area([[a, b], [c, b], [c, d], [a, d]], box, frame):
                            box.append([[a, b], [c, b], [c, d], [a, d]])
                            #cv2.rectangle(frame, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                        else:
                            global c_slide
                            # b1_area, b2_area = get_box_area(
                            #    c_slide, [[a, b], [c, b], [c, d], [a, d]])
                            b1_area = get_box_area(c_slide)
                            b2_area = get_box_area(
                                [[a, b], [c, b], [c, d], [a, d]])
                            if b2_area > b1_area:
                                box.remove(c_slide)
                                box.append([[a, b], [c, b], [c, d], [a, d]])
                                #cv2.rectangle(frame, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                                #cv2.rectangle(frame, (c_box[0][0], c_box[0][1]), (c_box[2][0], c_box[2][1]), color=(255, 0, 0), thickness=2)
                            #cv2.rectangle (frame,(a,b),(c,d), color=(0, 255, 0), thickness=2)
                            # plt.imshow(frame)
                            # plt.show()
    if verbose:
        print('     Total number of vehicle regions predicted in frame:', len(box))
    marked_boxes = []

    ## Build the final bounding boxes unified to same format (All the approaches)
    for rect in box:
        cv2.rectangle(frame, (rect[0][0], rect[0][1]), (rect[2][0], rect[2][1]), color=(
            0, 255, 0), thickness=2)
        marked_boxes.append(
            ((rect[0][0], rect[0][1]), (rect[2][0], rect[2][1])))

    if visualize_frames:
        # plt.imshow(frame)
        # plt.show()
        if verbose:
            print('     Visualising points and predicted frames')
        points_in_image(p, np.array(averages)[:, :2], color, frame)

    return frame, box


def points_in_image(points, averages, colouring, frame):
    """
        Function which can help in scattering the points over frame and visualize information of the points on hover.

        Parameters
        ----------
        :param points: Pointcloud data
        :param averages: Clustered and averaged points which are considered for region proposal
        :param colouring: Coloring of the points which are to be scattered. n_points should be equal to n_coloring values
        :param frame: The image frame on which points are to be scattered
    """

    frame_copy = np.copy(frame)
    fig, ax = plt.subplots()

    ## Scatter points based on transformed X & Y coordinates of Radar points and color based on its cluster value
    sc = ax.scatter(points[0, ], points[1, ], c=colouring[0], s=8, alpha=0.5)
    averages = np.transpose(averages)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    t = sc.get_offsets()

    def update_annot(ind):
        """
        Build the hover data
        """
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
        """
        Capture the hover event and perform suitable action(s)
        """
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

    ## Scatter the predicted moving vehicle predicted points
    sc2 = ax.scatter(averages[0, ], averages[1, ], s=14, alpha=0.9)
    
    plt.imshow(frame_copy)
    plt.show()


def get_box_area(box):
    """
        Helper function to calculate areas of box
        Parameters
        ----------
        :param box: Coordinates for first shape. Sample for rectangle [[a, b], [c, b], [c, d], [a, d]]

        Returns
        ----------
        box.area-> GEOSimpl : Area of both the boxes
    """
    return Polygon(box).area


def check_box_area(box1, boxes, frame, visualize=False):
    """
        Function checks if box1 is already present in the list of boxes. A box is considered to be present if 
        intersection area is greater than 85% for box which has been added with the one which is already present. 
        If Box1 has greater area, it is saved in global variable for replacing the other box which it intersected with.

        Parameters
        ----------
        :param box_1: Box cordinates which needs to checked if present already
        :param boxes: All the boxes which have been added prior to box_1 instance
        :param frame: The image frame over which the box rectangles need to visualised
        :param visualize: Boolean variable to check if user needs to visualize fully marked image frames 

        Returns
        ----------
        bool : If present already or not
    """
    for box2 in boxes:
        if not (box2 == box1):
            intersection = calculate_intersection(box1, box2)
            #a1, a2 = get_box_area(box1, box2)
            a1 = get_box_area(box1)
            a2 = get_box_area(box2)

            ## Checks if area of interesection between two boxes is less than 15% of other, if not it is considered as redundant prediction
            if intersection < 0.15*a1 and intersection < 0.15*a2:
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
    """
        Helper function to calculate Intersection over Union for two shapes
        Parameters
        ----------
        :param box_1: Coordinates for first shape. Sample for rectangle [[a, b], [c, b], [c, d], [a, d]]
        :param box_2: Coordinates for second shape. Sample for rectangle [[a, b], [c, b], [c, d], [a, d]]

        Returns
        ----------
        intersection -> list : Area of Intersection between two objects
    """
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    intersection = poly_1.intersection(
        poly_2).area  # / poly_1.union(poly_2).area
    return intersection


def calculate_iou(box_1, box_2):
    """
        Helper function to calculate Intersection over Union for two shapes
        Parameters
        ----------
        :param box_1: Coordinates for first shape. Sample for rectangle [[a, b], [c, b], [c, d], [a, d]]
        :param box_2: Coordinates for second shape. Sample for rectangle [[a, b], [c, b], [c, d], [a, d]]

        Returns
        ----------
        iou -> list : IOU value ranging between 0 to 1
    """
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def get_window_slides(frame, window_size, overlap):
    """
        Helper function to retrieve window co-ordinates for an image frame

        Parameters
        ----------
        :param frame: The image frame
        :param window_size: Size of windows which need to be extracted
        :param overlap: Overlapping which a window can have over other (0 -1) 

        Returns
        ----------
        window_slides -> list : List of window box coordinates
    """
    assert frame.shape[1] > window_size
    window_slides = []
    # print(frame.shape[0],frame.shape[1],window_size)

    ## Defines number of windows in rows and coloumns based on the frame shape, winodow size and overlap

    n_x_windows = int(frame.shape[1]//(window_size*overlap))
    n_y_windows = int(frame.shape[0]//(window_size*overlap))
    # print(n_x_windows,n_y_windows)
    
    ## Next row starting point
    y_window_seed = 0
    for i in range(0, n_y_windows):
        if (y_window_seed+window_size) < frame.shape[0]:
            
            ## Next column starting point
            x_window_seed = 0
            for j in range(0, n_x_windows):
                if (x_window_seed + window_size) < frame.shape[1]:
                    # print((x_window_seed,y_window_seed),(x_window_seed+window_size,y_window_seed+window_size))
                    window_slides.append(
                        [(x_window_seed, y_window_seed), (x_window_seed+window_size, y_window_seed+window_size)])
                    
                    ## Update column starting point
                    x_window_seed = int(x_window_seed + (window_size*overlap))
                else:
                    break
            
            ## Update row starting point
            y_window_seed = int(y_window_seed + (window_size*overlap))
        else:
            break
    return window_slides


def get_other_features(sub_frame):
    """
        Feature extractor function to extract the resized image bins and channel based histogram extracted 

        Parameters
        ----------
        :param sub_frame: The image frame

        Returns
        ----------
        rs_bins -> list : Binned resized image stored as list
        sf_hist -> list : The sub frame whose image channels features are extracted and stored as list
    """
    rs_bins = []
    sf_hist = []
    for i in range(3):
        rs_bins.append(cv2.resize(sub_frame[:, :, i], (32, 32)).ravel())
        sf_hist.append(np.histogram(sub_frame[:, :, i], bins=32))
    rs_bins = np.concatenate((rs_bins[0], rs_bins[1], rs_bins[2]))
    sf_hist = np.concatenate((sf_hist[0][0], sf_hist[1][0], sf_hist[2][0]))
    return rs_bins, sf_hist


def frame_slides_canvas(frame, slide_windows):
    """
        Function to draw rectangles over image frame

        Parameters
        ----------
        :param frame: The image frame  
        :param slide_windows: All the windows boxes which are to be drawn over the image frame as rectangles

        Returns
        ----------
        frame_copy -> ndarray : Image frame with rectangles drawn
    """
    frame_copy = np.array(frame)
    for slide_window in slide_windows:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        cv2.rectangle(frame_copy, (slide_window[0][0], slide_window[0][1]), (
            slide_window[1][0], slide_window[1][1]), (color), 1)
    return frame_copy


def predict_vehicles_slides_2(frame, method, slide_windows):
    """
        Function to predict all the windows with vehicle detections

        Parameters
        ----------
        :param frame: The sub-frame image region  
        :param method: Defines the SVM approach to follow. 0 for MODEL A approach and 1 for MODEL B approach
        :param slide_windows: All the windows boxes drawn for the original image frame

        Returns
        ----------
        vehicle_slides -> list : List of predicted vehicle boxes 
    """

    vehicle_slides = []

    ## Get the loaded model data
    global svc, xscaler

    for slide_window in slide_windows:

        sub_frame = frame[slide_window[0][1]: slide_window[1]
                          [1], slide_window[0][0]: slide_window[1][0], :]
        sub_frame = cv2.cvtColor(sub_frame, cv2.COLOR_RGB2YUV)
        sub_frame = cv2.resize(sub_frame, (64, 64))

        if method == 0:
            ## Get all the required features from images to feed in the classifer as input
            hog_feat = get_hog_features(sub_frame, 15, (8, 8))
            rs_bins, sf_hist = get_other_features(sub_frame)
            test_stacked = np.hstack(
                (rs_bins, sf_hist, hog_feat[0])).reshape(1, -1)
            #test_stacked = np.hstack((rs_bins, hog_feat[0])).reshape(1, -1)

            ## Normalize value using the Standard scaler value which is already built
            hog_feat_2 = xscaler.transform(test_stacked)
            # prediction=svc.predict(j)

            prediction = svc.predict(hog_feat_2)
        else:
            
            ## Extract the required image feature
            hog_feat = get_hog_features(sub_frame)

            prediction = svc.predict(hog_feat)
        if prediction == 1:
            vehicle_slides.append(slide_window)
    return vehicle_slides


def predict_vehicles_slides(frame, slide_windows):
    ## Replaced this function with predict_vehicles_slides_2 function
    vehicle_slides = []
    global svc
    for slide_window in slide_windows:

        sub_frame = frame[slide_window[0][1]: slide_window[1]
                          [1], slide_window[0][0]: slide_window[1][0], :]
        sub_frame = cv2.cvtColor(sub_frame, cv2.COLOR_RGB2YUV)
        sub_frame = cv2.resize(sub_frame, (64, 64))
        hog_feat = get_hog_features(sub_frame)
        # prediction=svc.predict(j)
        prediction = svc.predict(hog_feat)
        if prediction == 1:
            vehicle_slides.append(slide_window)
    return vehicle_slides


def get_hog_features(frame, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, feature_vector=True, multichannel=None):
    """
    Helper Function to call the hog feature extractor and return a single or two outputs based on visualize parameter

        Parameters
        ----------
        Same as HOG from skimage module

        Returns
        ----------
        Returns ravel list of normalized_blocks with hog features and its image if visualize param is set to True; Else returns just the former. 
    """
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
    """
        Function to check for the most overlapping area in the predicted car regions and return the final vehicle detection box

        Parameters
        ----------
        :param frame_size: Image frame size 1600 x 900 for all nuscenes image frames
        :param slide_windows: Sliding windows which are refined to vehicles predictions

        Returns
        ----------
        proba_frame -> Tuple : Refined windows take labelled as ndarray
        calculated_slides -> list : List of predicted vehicle boxes in Frame of size frame_size
    """

    ## Build a dummy frame based on original frame size
    proba_frame = np.zeros((frame_size[0], frame_size[1]))
    
    ## Increase counter value for all the predicted car regions
    for slide_window in slide_windows:
        proba_frame[slide_window[0][1]:slide_window[1][1],
                    slide_window[0][0]:slide_window[1][0]] += 1
    # print(proba_frame)

    ## Set all the counters to zero where the values are less than number of predicted car regions
    proba_frame[proba_frame <= (len(slide_windows)//2)] = 0

    proba_frame, n_vehicles = label(proba_frame)
    calculated_slides = []
    detected_slides = find_objects(proba_frame)

    for y_row, x_col in detected_slides:
        calculated_slides.append(
            ((x_col.start, y_row.start), (x_col.stop, y_row.stop)))
        #cv2.rectangle (frame,(x.start,y.start),(x.stop,y.stop), color=(0, 255, 0), thickness=1)
        # plt.imshow(frame)
        # plt.show()
    return proba_frame, calculated_slides


def save_video(frames, filename, fps, size):
    """
        Function to save as video in .avi format

        Parameters
        ----------
        :param frames: Takes in all the frames as list
        :param filename: Name of the file
        :param fps: Frames per second value
        :param size: Size of image frames
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = str('data/videos/')+filename+str(random.randint(0, 1000))+'.avi'
    video = cv2.VideoWriter(filename, fourcc, fps, (size[1], size[0]))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.destroyAllWindows()
    video.release()
    print(('Saved Video {} successfully').format(filename))


def get_annotations(nusc, scene_annotations, cam_token, visualize=False, verbose=False):
    """
        Function to get all the annotated object boxes

        Parameters
        ----------
        :param nusc: Nuscenes object
        :param scene_annotations: Scene annotation tokens
        :param cam_token: Camera sensor token
        :param visualize: Boolean variable to visualize all the annotated frame
        :param verbose: Boolean variable to display console logs

        Returns
        ----------
        list: Annotated boxes
    """
    annotated_boxes = []
    for ann_token in scene_annotations:
        cam = cam_token
        ann_record = nusc.get('sample_annotation', ann_token)


        ## Filtering the annotation to 'car' and 'truck', with a 'vehicle.moving' attribute
        if len(ann_record['attribute_tokens']) > 0 and ann_record['category_name'] in ['vehicle.car', 'vehicle.truck']:
            att_token = ann_record['attribute_tokens'][0]
            att_record = nusc.get('attribute', att_token)
            # ,'vehicle.stopped']):
            if(att_record['name'] in ['vehicle.moving']):
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(
                    cam_token, selected_anntokens=[ann_token])

                ## Build the annotated_boxes
                for box in boxes:

                    corners = view_points(
                        box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                    mins = corners.T.min(axis=0)
                    maxs = corners.T.max(axis=0)
                    a = int(mins[0])
                    b = int(mins[1])
                    c = int(maxs[0])
                    d = int(maxs[1])
                    annotated_boxes.append([[a, b], [c, b], [c, d], [a, d]])
                    if visualize:
                        if verbose:
                            print('     Visualising annotations')
                        frame = Image.open(data_path)
                        frame_copy = np.array(frame)
                        cv2.rectangle(frame_copy, (a, b),
                                      (c, d), (0, 255, 0), 2)
                        plt.imshow(frame_copy)
                        plt.show()
    return annotated_boxes


def str2bool(v):
    """
        Returns boolean value for the string
        Parameters
        ----------
        :param v: Value that needs to be checked and converted

        Returns
        ----------
        Boolean value for the inputted value

    """
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/31347222
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_accuracy(marked_boxes, annotated_boxes, frame, visualize=False, verbose=False):
    """
        Evaluation/Validation function to calculate the accuracy
        Parameters
        ----------
        :param marked_boxes: list -> Marked object boxes for the image
        :param annotated_boxes: list -> Annotated object boxes for the image
        :param frame: PIL Image -> frame taken for detection in the current instance
        :param visualize: Boolean variable to visualize object-by-object predicted vs annotated truth for comparison
        :param verbose: Boolean variable to display console logs

        Returns
        ----------
        Precision: float
        Recall: float
        True positives : int
        False Positives : int
    """
    tp = fp = fn = 0
    iou_list = []
    average = 0.5

    for annotated_box in annotated_boxes:
        frame_copy = np.copy(frame)

        ## Default values
        max_iou = -1
        pos = -1
        for i, marked_box in enumerate(marked_boxes):
            frame_copy2 = np.copy(frame_copy)
            iou = calculate_iou(marked_box, annotated_box)

            ## Checks for the best predicted/marked box match in comparison with annotated box
            if iou > max_iou and iou > 0.5:
                max_iou = iou
                pos = i
            """ cv2.rectangle(frame_copy2, (marked_box[0][0], marked_box[0][1]), (marked_box[2][0], marked_box[2][1]), color=(
            255, 0, 0), thickness=2)
            cv2.rectangle(frame_copy2, (annotated_box[0][0], annotated_box[0][1]), (annotated_box[2][0], annotated_box[2][1]), color=(
            0, 255, 0), thickness=2)
            plt.imshow(frame_copy2)
            plt.show() """
        if verbose:
            print('         IoU is:', max_iou)
        
        ## Build confusion matrix quadrants based on the 'max_iou' value
        if max_iou > 0.5:
            tp = tp + 1
        elif max_iou >= 0:
            fp = fp + 1
        if pos == -1:
            fn = fn + 1
        #print("Correct prediction",tp,"Wrong prediction",fp,"Not Predicted",fn)
        if max_iou >= -1:
            if visualize:
                if verbose:
                    print('         Visualising IOU taken vs actual')
                cv2.rectangle(frame_copy, (marked_boxes[pos][0][0], marked_boxes[pos][0][1]), (marked_boxes[pos][2][0], marked_boxes[pos][2][1]), color=(
                    255, 0, 0), thickness=2)
                cv2.rectangle(frame_copy, (annotated_box[0][0], annotated_box[0][1]), (annotated_box[2][0], annotated_box[2][1]), color=(
                    0, 255, 0), thickness=2)
                cv2.putText(frame_copy, str(round(max_iou, 3)), (marked_boxes[pos][0][0], marked_boxes[pos][0]
                                                                 [1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                plt.imshow(frame_copy)
                plt.show()
            iou_list.append(max_iou)

    if tp > 0 or fp > 0:
        precision = tp / (len(marked_boxes))
    else:
        precision = 0
    if tp > 0 or fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if len(iou_list) > 0.1:
        average = round(sum(iou_list) / len(iou_list), 3)
    # if verbose:
    #    print('     Average IoU is:', average)

    return precision, recall, tp, fp


def run_detection_system(method=(2, 0), validate_results=False, visualize_frames=False, visualize_sub_frames=False, verbose=False, save_file=False):
    """
        Main function which takes uses all the helper functions to make the detections
        Parameters
        ----------
        :param method: Tuple which specifies the (classifier,isParallel)
        :param validate_results: Boolean variable to check if user needs to validate results
        :param visualize_frames: Boolean variable to check if user needs to visualize fully marked image frames
        :param visualize_sub_frames: Boolean variable to check if user needs to visualize region frames which are proposed and marked
        :param verbose: Boolean variable to display console logs
        :param save_file: Boolean variable to save detections to a file
    """
    ## Load Nuscenes object and specify required channels
    location = 'data/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=location, verbose=False)
    pointsensor_channel = 'RADAR_FRONT'
    camera_channel = 'CAM_FRONT'
    frames = []
    global net, output_layers, classes, svc, xscaler

    ## Loading model/network once per session, so that it is not repeated for every single scene/frame
    if method[0] > 1:
    
        net, output_layers, classes = load_net()
        if verbose:
            print('Loaded YOLO Net')
        filename = 'YOLOv3_'
    else:
        if method[0] == 0:
            svc, xscaler = load_svc_2()
        else:
            svc = load_svc()
        if verbose:
            print('Loaded SVM predictor')
        filename = 'HOG_SVM_'

    t0 = time.time()

    ## Scenes iterator
    for scene in nusc.scene:
        # if verbose:
        #    print('Scene description: ',scene['description'])
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        check_token = first_sample_token
        pre = []
        rec = []

        while (check_token != '') and scene['name'] == 'scene-0061':
            if verbose:
                print('     -------------------New-Scene----------------')
            sample_record = nusc.get('sample', check_token)

            ## Getting front radar and camera sensors' token value
            pointsensor_token = sample_record['data'][pointsensor_channel]
            camera_token = sample_record['data'][camera_channel]

            ## Get all the frames with detected moving vehicles
            marked_frames, marked_boxes = get_marked_frames(
                nusc, pointsensor_token, camera_token, method, visualize_frames, visualize_sub_frames, verbose)
            frames.append(marked_frames)

            ## Validates the prediction based on  validate_result parameter
            if validate_results:
                scene_annotations = sample_record['anns']
                annotated_boxes = get_annotations(
                    nusc, scene_annotations, camera_token)
                cam = nusc.get('sample_data', camera_token)
                frame = Image.open(osp.join(nusc.dataroot, cam['filename']))
                precision, recall, trueP, trueN = get_accuracy(
                    marked_boxes, annotated_boxes, frame, visualize_sub_frames, verbose)
                pre.append(precision)
                rec.append(recall)
            check_token = sample_record['next']

        if validate_results and scene['name'] == 'scene-0061':
            print('Avg Precision is:', sum(pre) / (len(pre)))
            print('Avg Recall is:', sum(rec)/(len(rec)))
            ## Not using mAP for just one scene and hence commented the below function call
            #getmap(pre, rec)
    t1 = time.time()
    t = t1-t0
    print('Time for ', filename, 'is:', t)
    ## Save the detected frames as video if needed
    if save_file:
        save_video(frames, filename, 10, frames[0].shape[:2])


def validate_args(args):
    """
        Validates if the arguments passed are correct
        :param args: Arguments retrieved through argsparser
    """

    try:
        if type(args.c) == int and type(args.p) == int and args.c >= 0 and args.c <= 3 and args.p >= 0 and args.p <= 1:
            if args.c > 1 and args.p > 0:
                print(
                    'No Parallel processing required for YOLOv3 version. Setting it to default')
                args.p = 0
        else:
            raise ValueError(
                'Use -h for help. You have entered a wrong integer input')
        if type(args.t) != bool and type(args.f) != bool and type(args.s) != bool and type(args.v) != bool:
            raise ValueError(
                'Use -h for help. You have entered a wrong boolean input')

    except Exception as error:
        print('Caught this error: ' + repr(error))
        exit()
    return args


if __name__ == "__main__":
    # run_detection_system((2, 0), True,
    #                     False, False, False, False)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=2,
                        help="0 = Slow SVM, 1 = Fast SVM, 2 = Modified YOLO, 3 = Orginal YOLO")
    parser.add_argument('-p', type=int, default=0,
                        help="0 = For normal processing, 1 = For parallel processing")
    parser.add_argument('-t', type=str2bool, nargs='?',
                        const=True, default=False, help="Validate results")
    parser.add_argument('-f', type=str2bool, nargs='?',
                        const=True,  default=False, help="Visualize Frames")
    parser.add_argument('-s', type=str2bool, nargs='?',
                        const=True,  default=False, help="Visualize Sub-Frames")
    parser.add_argument('-v', type=str2bool, nargs='?',
                        const=True, default=False, help="Verbose")
    parser.add_argument('-k', type=str2bool, nargs='?',
                        const=True,  default=False, help="Save/keep detections to a file")

    args = parser.parse_args()
    args = validate_args(args)

    run_detection_system((args.c, args.p), args.t,
                         args.f, args.s, args.v, args.k)
