import sys
import numpy as np
from copy import deepcopy
sys.path.append("..")
import lib.label_map_util
import datetime
import tensorflow as tf

import pygame

pygame.font.init()

'''
------ y1,x1 
|          |
|          |
|          |
y2,x2 ------
'''


class Net:
    def __init__(self, graph_fp, labels_fp, num_classes=90, threshold=0.6):
        self.graph_fp = graph_fp
        self.labels_fp = labels_fp
        self.num_classes = num_classes

        self.graph = None
        self.label_map = None
        self.categories = None
        self.category_index = None

        self.bb = None
        self.bb_origin = None
        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None

        self.in_progress = False
        self.session = None
        self.threshold = threshold
        self._load_graph()
        self._load_labels()
        self._init_predictor()

    def _load_labels(self):
        self.label_map = lib.label_map_util.load_labelmap(self.labels_fp)
        self.categories = lib.label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = lib.label_map_util.create_category_index(self.categories)

    def _load_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        tf.get_default_graph().finalize()

    def _display(self, filtered_results, processed_img, display_img, fps=0):
        h, w, _ = processed_img.shape
        h_dis, w_dis, _ = display_img.shape
        ratio_h = h_dis / h
        ratio_w = w_dis / w
        
        size = (w_dis, h_dis)
        display = pygame.display.set_mode(size)
        snapshot = pygame.surfarray.make_surface(display_img)      
        
        font_size = 30
        font_color = (0, 255, 255) #R G B: Light Blue
        font = pygame.font.Font(None, font_size)   
        
        line_thickness = 3
        line_color = (255, 0, 0) #R G B: Red
        
        for res in filtered_results:
            y1, x1, y2, x2 = res["bb_o"]
            y1, y2 = int(y1 * ratio_h), int(y2 * ratio_h)
            x1, x2 = int(x1 * ratio_w), int(x2 * ratio_w)
            
            points = [(y1, x1), (y1, x2), (y2, x2), (y2, x1)]
            pygame.draw.lines(snapshot, line_color, True, points, line_thickness)
            
            text = font.render(res["class"], True, font_color) 
            text = pygame.transform.rotate(text, 90)           
            snapshot.blit(text, (y1, x1+3))
        
        snapshot = pygame.transform.rotate(snapshot, 270)
        
        if fps > 0:
            font_size = 50
            font_color = (99, 195, 206) #R G B: Aquamarine
            font = pygame.font.SysFont("freesansbold", font_size) 
            text = font.render('FPS: %s' % fps, True, font_color)            
            snapshot.blit(text, (10, 30))
        
        display.blit(snapshot, (0,0))        
        pygame.display.flip()

    def _init_predictor(self):
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def predict(self, img, display_img, mode="camera"):
        self.in_progress = True
        start = datetime.datetime.now().microsecond * 0.001

        with self.graph.as_default():
            print ('[INFO] Read the image ..')
            img_copy = deepcopy(img)           
            height, width, _ = img.shape
            print ('[INFO] Shape of this image is -- [height: %s, width: %s]' % (height, width))

            image_np_expanded = np.expand_dims(img, axis=0)

            print ('[INFO] Detecting objects ...')
            session_start = datetime.datetime.now().microsecond * 0.001
            (boxes, scores, classes, num_detections) = self.session.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            session_end = datetime.datetime.now().microsecond * 0.001
            print ('[INFO] Filtering results ...')
            filtered_results = []
            
            max_objects = 10
            
            for i in range(max_objects):
                score = scores[0][i]
                if score >= self.threshold:
                    y1, x1, y2, x2 = boxes[0][i]
                    y1_o = int(y1 * height)
                    x1_o = int(x1 * width)
                    y2_o = int(y2 * height)
                    x2_o = int(x2 * width)
                    predicted_class = self.category_index[classes[0][i]]['name']
                    filtered_results.append({
                        "score": score,
                        "bb": boxes[0][i],
                        "bb_o": [y1_o, x1_o, y2_o, x2_o],
                        "img_size": [height, width],
                        "class": predicted_class
                    })
                    print ('[INFO] %s: %s' % (predicted_class, score))
            
            if(mode == "camera"):
                end = datetime.datetime.now().microsecond * 0.001
                elapse = end - start
                fps = np.round(1000.0 / elapse, 3)
                session_elapse = session_end - session_start
                sfps = np.round(1000.0 / session_elapse, 3)
                print ('+++++++++++++++++++++++ SFPS: ', sfps)
                print ('----------------------- FPS: ', fps)
                self._display(filtered_results, processed_img=img_copy, display_img=display_img, fps=fps)
            elif(mode == "static"):
                self._display(filtered_results, processed_img=img_copy, display_img=display_img)
                
        self.in_progress = False

    def get_status(self):
        return self.in_progress

    def kill_predictor(self):
        self.session.close()
        self.session = None
