import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import sys
import numpy as np
import onnxruntime as ort
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from ray import serve
import warnings
warnings.filterwarnings("ignore")

import sys
YOLO_DIR_PATH = "MLmodels/singleFace"
sys.path.append("library/DeepSORT_YOLOv5_Pytorch/") #path_to_yolov5_repository
from yolov5.utils.general import xyxy2xywh, non_max_suppression
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.datasets import letterbox
    
def letterbox_new(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup and r>1:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)
    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    bottom = int(round(dh + 0.1))
    right = int(round(dw + 0.1)) 
    left, top =0, 0
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def resize(img, new_shape, letterboxing):
    """_summary_

    Args:
        img (_type_): The input image on which we run the yolo model 
        new_shape (_type_): The output shape to which the give input image needs to be resized 
        letterboxing (_type_): Boolean value which determines whether we want to resize the image maintaining aspect ratio or not

    Returns:
        _type_: Resizes the given image to yolo input that is of shape (416, 416)
    """
    img_c = img.copy()
    shape = img.shape[:2]
    if letterboxing:
        img_c = letterbox(img_c, new_shape= new_shape, scaleup = True, auto=False)[0]
    
    img_c = img_c[:, :, ::-1].transpose(2, 0, 1)
    img_c = np.ascontiguousarray(img_c, dtype=np.float32)
    img_c = torch.from_numpy(img_c)
    if img.ndim == 3:
        img_c = img_c.unsqueeze(0)
    img_c = img_c/255.
    return img_c

def clip_boxes(boxes, shape):
    """ Clip boxes (xyxy) to image shape (height, width)

    Args:
        boxes (_type_): The bounding box information of the detection
        shape (_type_): The shape of the original image on which the yolo model is ran
        
    """
    
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def loadCustomYoloModel(weights, classes):
    """_summary_

    Args:
        weights (_type_): the onnx weight file of the yolo model
        classes (_type_): the classes which we want to detect on an image

    Returns:
        _type_: takes the yolo weights and classes as parameters and returns the loaded model 
    """
    yolomodel = ort.InferenceSession(weights, providers=['CPUExecutionProvider'])
    classes_available = {0:'person',1:'face',2:'vehicle',3:'animal'}
    class_names = {}
    for i in classes:
        class_names[i] = classes_available[i]
    return yolomodel, class_names

def run_yolo_model(im0, weight, img_size = 416, load_classes = [1]):
    """_summary_

    Args:
        im0 (_type_): The input image on which we intend to run the yolo model 
        img_size (int, optional): The image size which yolo takes as input. Defaults to 416.
        load_classes (list, optional): The classes which we want to detect, can be modified for other class detections. Defaults to [1].

    Returns:
        _type_: Given an input image, the yolo model runs on it and returns the detections of shape (n, 6) 
        where n is number of detections
    """
    
    if isinstance(img_size, int):
        new_img_size = (img_size, img_size)
    else:
        new_img_size = img_size
    img = resize(im0, new_shape = new_img_size, letterboxing = True)
    
    s = '%gx%g ' % img.shape[2:]   

    model, class_names = loadCustomYoloModel(weight, load_classes)

    with torch.no_grad():
        pred = model.run(None, {"images": img.cpu().numpy()})[0]
        
    # Apply NMS and filter object other than person (cls:0)
    pred_torch = torch.from_numpy(pred) 
    pred = non_max_suppression(pred_torch, conf_thres = 0.5, iou_thres = 0.2, classes= list(class_names.keys()), agnostic= False)
    # get all obj ************************************************************
    det = pred[0]  
    
    # to scale up the bounding boxes to the original image size
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
        tmp_h = (bbox_xywh[..., 2] * bbox_xywh[..., 3]) > 0.0 
        det = det[tmp_h]
        bbox_xywh = bbox_xywh[tmp_h]
        return det
    else:
        return det

def get_face_crops(face_bbox_array, image):
    x1,y1,x2,y2 = int(face_bbox_array[0]), int(face_bbox_array[1]), int(face_bbox_array[2]), int(face_bbox_array[3])
    face_crops = image[y1:y2, x1:x2]
    return face_crops

def loadFacenet(prototx_path, caffemodel_path):
    """given the prototxt and caffemodel path, it loads the mobilefacenet model

    Args:
        prototx_path (_type_): path to .prototxt file
        caffemodel_path (_type_): path to .caffemodel

    Returns:
        _type_: returns the loaded mobilefacenet model
    """
    opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt= prototx_path, caffeModel = caffemodel_path)
    return opencv_dnn_model

def resizeFaceStandardize(img, new_shape, letterboxing):
    """_summary_

    Args:
        img (_type_): image that needs to be resized 
        new_shape (_type_): the shape to which the input images needs to be resized
        letterboxing (_type_): boolean to determine whether to resize maintaining aspect ratio or not 

    Returns:
        _type_: returns a resized image of shape (batch_size, num_channels, width, height)
    """
    img_c = img.copy()
    # img_c = img_c[:, :, 
    shape = img.shape[:2]
    if letterboxing:
        img_c = letterbox_new(img_c, new_shape= new_shape, scaleup = True, auto=False)[0]
    else:
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img_c = cv2.resize(img_c, dsize=(64, 128), interpolation = interp)
        
    img_c = img_c[:, : , ::].transpose(2, 0, 1)     # converts to bgr
    
    img_c = np.ascontiguousarray(img_c, dtype=np.float32)
    img_c = torch.from_numpy(img_c)
    if img.ndim == 3:
        img_c = img_c.unsqueeze(0)       # adds batch size to the reshaped array --> (1,3,112,96)
    img_c = (img_c - 127.5)/128.
    return img_c

def getLetterBoxFaceEmbeddingVector(cropped_image, prototxt_model, caffe_model, new_shape =(112, 96)):
    """ takes a cropped image and returns an embedding vector 

    Args:
        cropped_image (_type_): cropped image that we get from yolo model (face in this case)
        new_shape (tuple, optional): _description_. Defaults to (112, 96).
        prototxt_model (str, optional): _description_. Defaults to '/home/harsh/harsha/56-camera-ai-pipeline/mobileFacenet/modifiedFacenet.prototxt'.
        caffe_model (str, optional): _description_. Defaults to '/home/harsh/harsha/56-camera-ai-pipeline/mobileFacenet/MobileFaceNet.caffemodel'.

    Returns:
        _type_: returns the embedding vector
    """
    facenet_model = loadFacenet(prototxt_model, caffe_model)
    img = resizeFaceStandardize(cropped_image, new_shape = new_shape, letterboxing = True)
    facenet_model.setInput(img.numpy())
    embedding_vector = facenet_model.forward()
    return embedding_vector

def get_embedding_vector(image):
    # if os.path.exists(args.image_path):
    #     image = cv2.imread(args.image_path)
    # else:
    #     image = args.image_path
    # print("image" ,type(image))
    
    bbox_info = run_yolo_model(image, f"{YOLO_DIR_PATH}/yolov5s_0.0.18.onnx")
    
    face_crop = get_face_crops(bbox_info[0, :4], image)
    embedding_vector = getLetterBoxFaceEmbeddingVector(
        face_crop,
        f"{YOLO_DIR_PATH}/mobileFacenet/modifiedFacenet.prototxt",
        f"{YOLO_DIR_PATH}/mobileFacenet/MobileFaceNet.caffemodel"
    )
    return embedding_vector
    

# def parse_args():
#     """change the default paths to the models below and give the image path at run time 

#     Returns:
#         _type_: _description_
#     """
#     parser = argparse.ArgumentParser(description = "given an image path, it computes the embedding vector")
#     # parser.add_argument("--yolo-weight", default = f"{YOLO_DIR_PATH}/yolov5s_0.0.18.onnx", help = "path to yolo model weight")
#     # parser.add_argument("--prototxt", default = f"{YOLO_DIR_PATH}/mobileFacenet/modifiedFacenet.prototxt", help = "path to prototxt model weight")
#     # parser.add_argument("--caffe", default = f"{YOLO_DIR_PATH}/mobileFacenet/MobileFaceNet.caffemodel", help = "path to caffe model weight")
#     parser.add_argument("--image-path", default = "", help = "path to the image")
#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     global args
#     args = parse_args()
#     print(args)
#     emb_vector = get_embedding_vector(args)
#     np.save("file", emb_vector)

