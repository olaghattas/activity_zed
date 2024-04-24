import rclpy
from rclpy.node import Node
import rclpy
from rclpy.node import Node
import numpy as np 
from std_msgs.msg import Int32
import cv2
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model import LSTM
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_hidden = 128
n_joints = 25*2
n_categories = 2
n_layer = 3
seq_len = 30
model = LSTM(n_joints,n_hidden,n_categories,n_layer, seq_len).to(device)


model.load_state_dict(torch.load('rnn_train_1_dec7.pth'))
#model.load_state_dict(torch.load('model_train_1_dec27.pth'))

model.eval()

pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
def extract_keypoints(image_rgb): 
    try:
        results = pose.process(image_rgb)
        landmarks=results.pose_landmarks.landmark
    except Exception as e:
        # print('Error file=', fn)
        # print('Error=', e)
        return None
    xys=[]
    for landmark in landmarks:
        xys.append([landmark.x, landmark.y])
    xys=np.array(xys)
    return xys

class ActivityNode(Node):

    def __init__(self):
        super().__init__('ActivityNode')
        
        self.publisher_ = self.create_publisher(Int32, '/person_eating', 10)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Replace with your actual image topic
            self.image_callback,
            10  # QoS profile
        )
        self.cv_bridge = CvBridge()
        self.queue = deque(maxlen=seq_len)
        self.view=True
        self.count = 0
        self.total = 0
 

    def image_callback(self, msg):
        
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_rgb=cv_image
        xys=extract_keypoints(image_rgb)
        try:
            xys=xys[:25].ravel() #first 25 keypoints
                
            self.queue.append(xys)

            pred=-1
            if len(self.queue) == seq_len:
                # print('time to predict')
                action=np.array(self.queue)
                tx=torch.from_numpy(action).float()
                tx=tx.unsqueeze(0).to(device)
                output = model(tx).cpu().detach().numpy()
                po=output.argmax(axis=1)
                pred=po[0]
                self.count = self.count+1
                self.total = self.total+pred
                #print("count:",self.count)
                 

                if(self.count>20):
                    average = self.total/self.count
                    if(average>.60):
                        msg = Int32
                        msg.data = 1
                        self.count = 0
                        self.total = 0
                        print("average>60:",msg.data)
                        self.publisher_.publish(msg)
                        
                        
                    else:
                        self.count = 0
                        self.total = 0
                        msg = Int32
                        msg.data = 0
                        print("average<60:",msg.data)
                        self.publisher_.publish(msg)
                

                if self.view:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 50)
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv_image = cv2.putText(cv_image, "pred="+str(pred), org, font,  
                            fontScale, color, thickness, cv2.LINE_AA)
                    cv2.imshow('Activity Recognition', cv_image)
                    cv2.waitKey(1)
                
        except Exception as e:  # Specify the exception type if possible
            #print("I am here!")
            pass


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ActivityNode() 
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

