#!/usr/bin/env python3


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
import torchvision
from time import time


class face_person_mobilenet:


    def __init__(self):

        # print(os.getcwd())

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)

        backbone = mobilenet.features

        backbone.out_channels = 576

        anchor_generator = AnchorGenerator(sizes=((31,64,128,256,512),), aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        self.model = FasterRCNN(backbone, 3, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

        print('Loading model state...')

        self.model.load_state_dict(torch.load('src/test_lidar/model/mobilenet_v3_state_dict_pytorch.pth'))

        print('Finished loading model state.')

        self.model.to(self.device)

        print("Model is running on {}".format(self.device))

        self.model.eval()
        
        self.labels_dict =  {1:'Person', 2:'Face'}
        self.colors_dict = {1:(0,255,0), 2:(0,0,255)}


    def simple_detection(self, img):
        # Preprocessing the input image
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor_normal = image_tensor/255

        # Passing preprocessed image into model
        output = self.model([image_tensor_normal])

        # Returns the bounding boxes, labels for each bounding box, and the prediction probabilities for each box
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        return boxes.detach().cpu().numpy(), labels.detach().cpu().numpy(), scores.detach().cpu().numpy()


class coco_mobilenet:


    def __init__(self):

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

        self.model.to(self.device)

        print("Model is running on {}".format(self.device))

        self.model.eval()

        self.category_map, self.category_index = self.label_dict()

    
    def label_dict(self):

        category_map = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            5: 'airplane',
            6: 'bus',
            7: 'train',
            8: 'truck',
            9: 'boat',
            10: 'traffic light',
            11: 'fire hydrant',
            13: 'stop sign',
            14: 'parking meter',
            15: 'bench',
            16: 'bird',
            17: 'cat',
            18: 'dog',
            19: 'horse',
            20: 'sheep',
            21: 'cow',
            22: 'elephant',
            23: 'bear',
            24: 'zebra',
            25: 'giraffe',
            27: 'backpack',
            28: 'umbrella',
            31: 'handbag',
            32: 'tie',
            33: 'suitcase',
            34: 'frisbee',
            35: 'skis',
            36: 'snowboard',
            37: 'sports ball',
            38: 'kite',
            39: 'baseball bat',
            40: 'baseball glove',
            41: 'skateboard',
            42: 'surfboard',
            43: 'tennis racket',
            44: 'bottle',
            46: 'wine glass',
            47: 'cup',
            48: 'fork',
            49: 'knife',
            50: 'spoon',
            51: 'bowl',
            52: 'banana',
            53: 'apple',
            54: 'sandwich',
            55: 'orange',
            56: 'broccoli',
            57: 'carrot',
            58: 'hot dog',
            59: 'pizza',
            60: 'donut',
            61: 'cake',
            62: 'chair',
            63: 'couch',
            64: 'potted plant',
            65: 'bed',
            67: 'dining table',
            70: 'toilet',
            72: 'tv',
            73: 'laptop',
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell phone',
            78: 'microwave',
            79: 'oven',
            80: 'toaster',
            81: 'sink',
            82: 'refrigerator',
            84: 'book',
            85: 'clock',
            86: 'vase',
            87: 'scissors',
            88: 'teddy bear',
            89: 'hair drier',
            90: 'toothbrush'
        }

        category_index = {
            1: {'id': 1, 'name': 'person'},
            2: {'id': 2, 'name': 'bicycle'},
            3: {'id': 3, 'name': 'car'},
            4: {'id': 4, 'name': 'motorcycle'},
            5: {'id': 5, 'name': 'airplane'},
            6: {'id': 6, 'name': 'bus'},
            7: {'id': 7, 'name': 'train'},
            8: {'id': 8, 'name': 'truck'},
            9: {'id': 9, 'name': 'boat'},
            10: {'id': 10, 'name': 'traffic light'},
            11: {'id': 11, 'name': 'fire hydrant'},
            13: {'id': 13, 'name': 'stop sign'},
            14: {'id': 14, 'name': 'parking meter'},
            15: {'id': 15, 'name': 'bench'},
            16: {'id': 16, 'name': 'bird'},
            17: {'id': 17, 'name': 'cat'},
            18: {'id': 18, 'name': 'dog'},
            19: {'id': 19, 'name': 'horse'},
            20: {'id': 20, 'name': 'sheep'},
            21: {'id': 21, 'name': 'cow'},
            22: {'id': 22, 'name': 'elephant'},
            23: {'id': 23, 'name': 'bear'},
            24: {'id': 24, 'name': 'zebra'},
            25: {'id': 25, 'name': 'giraffe'},
            27: {'id': 27, 'name': 'backpack'},
            28: {'id': 28, 'name': 'umbrella'},
            31: {'id': 31, 'name': 'handbag'},
            32: {'id': 32, 'name': 'tie'},
            33: {'id': 33, 'name': 'suitcase'},
            34: {'id': 34, 'name': 'frisbee'},
            35: {'id': 35, 'name': 'skis'},
            36: {'id': 36, 'name': 'snowboard'},
            37: {'id': 37, 'name': 'sports ball'},
            38: {'id': 38, 'name': 'kite'},
            39: {'id': 39, 'name': 'baseball bat'},
            40: {'id': 40, 'name': 'baseball glove'},
            41: {'id': 41, 'name': 'skateboard'},
            42: {'id': 42, 'name': 'surfboard'},
            43: {'id': 43, 'name': 'tennis racket'},
            44: {'id': 44, 'name': 'bottle'},
            46: {'id': 46, 'name': 'wine glass'},
            47: {'id': 47, 'name': 'cup'},
            48: {'id': 48, 'name': 'fork'},
            49: {'id': 49, 'name': 'knife'},
            50: {'id': 50, 'name': 'spoon'},
            51: {'id': 51, 'name': 'bowl'},
            52: {'id': 52, 'name': 'banana'},
            53: {'id': 53, 'name': 'apple'},
            54: {'id': 54, 'name': 'sandwich'},
            55: {'id': 55, 'name': 'orange'},
            56: {'id': 56, 'name': 'broccoli'},
            57: {'id': 57, 'name': 'carrot'},
            58: {'id': 58, 'name': 'hot dog'},
            59: {'id': 59, 'name': 'pizza'},
            60: {'id': 60, 'name': 'donut'},
            61: {'id': 61, 'name': 'cake'},
            62: {'id': 62, 'name': 'chair'},
            63: {'id': 63, 'name': 'couch'},
            64: {'id': 64, 'name': 'potted plant'},
            65: {'id': 65, 'name': 'bed'},
            67: {'id': 67, 'name': 'dining table'},
            70: {'id': 70, 'name': 'toilet'},
            72: {'id': 72, 'name': 'tv'},
            73: {'id': 73, 'name': 'laptop'},
            74: {'id': 74, 'name': 'mouse'},
            75: {'id': 75, 'name': 'remote'},
            76: {'id': 76, 'name': 'keyboard'},
            77: {'id': 77, 'name': 'cell phone'},
            78: {'id': 78, 'name': 'microwave'},
            79: {'id': 79, 'name': 'oven'},
            80: {'id': 80, 'name': 'toaster'},
            81: {'id': 81, 'name': 'sink'},
            82: {'id': 82, 'name': 'refrigerator'},
            84: {'id': 84, 'name': 'book'},
            85: {'id': 85, 'name': 'clock'},
            86: {'id': 86, 'name': 'vase'},
            87: {'id': 87, 'name': 'scissors'},
            88: {'id': 88, 'name': 'teddy bear'},
            89: {'id': 89, 'name': 'hair drier'},
            90: {'id': 90, 'name': 'toothbrush'}
        }

        return category_map, category_index


    def simple_detection(self, img):
        # Preprocessing the input image
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor_normal = image_tensor/255

        # Passing preprocessed image into model
        output = self.model([image_tensor_normal])

        # Returns the bounding boxes, labels for each bounding box, and the prediction probabilities for each box
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        return boxes.detach().cpu().numpy(), labels.detach().cpu().numpy(), scores.detach().cpu().numpy()


class coco_segmentation_mobilenet:


    def __init__(self):

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)

        self.model.to(self.device)

        print("Model is running on {}".format(self.device))

        self.model.eval()

        self.category_map = self.label_dict()

        self.trf = T.Compose([T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


    def label_dict(self):

        label_dict = {
                    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                    }

        return label_dict


    def decode_segmap(self, image, orig_image, nc=21):
        # Source for this code: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

        label_colors = np.array([  # 0=background
                (0,0,0),
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 2]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 0]
        bgr = np.stack([b, g, r], axis=2)
        bgr = np.where(bgr==[0,0,0], orig_image, bgr)
        return bgr


    def simple_segmentation(self, img):

        pil_img = Image.fromarray(img)
        inp = self.trf(pil_img).unsqueeze(0).to(self.device)
        out = self.model(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        bgr = self.decode_segmap(om, img)

        # composition = rgb + img

        return bgr