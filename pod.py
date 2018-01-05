import sys
from models import object_detection
import tensorflow as tf

import pygame
import pygame.camera



model_name = "ssd_mobilenet_v1_coco_2017_11_17"
num_classes = 90
threshold = 0.6

net = object_detection.Net(graph_fp='%s/frozen_inference_graph.pb' % model_name,
                           labels_fp='data/label.pbtxt',
                           num_classes=num_classes,
                           threshold=threshold)


def detect(mode="camera"):    
    if mode == "static":
        img_fp = 'test_images/3.jpg'  #Image file to be used
        snapshot = pygame.image.load(img_fp)                
        snapshot = pygame.transform.rotate(snapshot, 90)
        img = pygame.surfarray.array3d(snapshot)
        net.predict(img=img, display_img=img, mode="static")
        input("Press ENTER to exit...")
        
    elif mode == "camera":
        size = (640,480)  #Image size to be captured
        pygame.camera.init()
        cam = pygame.camera.Camera("/dev/video0", size)
        cam.start()
        display = pygame.display.set_mode(size)
        pygame.display.set_caption("Pygame Object Detection - Press ESCAPE to exit")
        
        
        """ We have to rotate the images because of a weird bug on surfarray.array3d that create an image
        rotated from the original surface. The rotating has to be done before and after detection, being 90°
        before it and 270° after it, this way we can reconstruct the original image """       
        
        image_size = 320  #Image size to be used by the network

        while True:
            snapshot = cam.get_image()
            snapshot = pygame.transform.rotate(snapshot, 90)
            frame = pygame.surfarray.array3d(snapshot)
            resize_frame = pygame.surfarray.array3d(pygame.transform.scale(snapshot, (image_size, image_size)))
                
            in_progress = net.get_status()
            if (not in_progress):
                net.predict(img=resize_frame, display_img=frame)
            else:
                print ('[Warning] drop frame or in progress')
            events = pygame.event.get()
            for e in events:
                if (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    cam.stop()
                    sys.exit()

if __name__ == '__main__':
    detect(mode="static")
    #detect(mode="camera")
    
