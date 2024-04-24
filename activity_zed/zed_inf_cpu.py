import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from zed_interfaces.msg import ObjectsStamped
import cv2
from cv_bridge import CvBridge
import os
import rclpy
from rclpy.node import Node
 
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import torch.optim as optim


class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num, seq_len):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim)
        self.bn = nn.BatchNorm1d(seq_len)
        
    def forward(self,inputs):
        x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out
    
 

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_hidden = 128
n_joints = 25*2
n_categories = 2
n_layer = 3
seq_len = 300
model = LSTM(n_joints,n_hidden,n_categories,n_layer, seq_len) #.to(device)

path = os.getenv("path_activity")
# ckpt='lstm_zed_april15.pth'
ckpt= path + 'activity_zed/lstm_zed_april17_cpu.pth'

print('Loading ckpt: ', ckpt)
model.load_state_dict(torch.load(ckpt))
model.eval()
print('Model loaded')

view_img=False
print('view_img?=', view_img)


class ZedInf(Node):

    def __init__(self):
        super().__init__('zed_sub')
        self.publisher_ = self.create_publisher(String, '/person_eating', 10)
        self.subscription = self.create_subscription(
            Image,
            '/zed_dining_room/zed_node_dining_room/left_raw/image_raw_color',
            self.image_callback,
            10)
        
        self.sub_skeleton = self.create_subscription(
            ObjectsStamped,
            'zed_dining_room/zed_node_dining_room/body_trk/skeletons',
            self.skeleton_callback,
            10)


        self.bridge = CvBridge()

        self.kp_indices=[1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17, 30,31,32,33,34,35,36,37]
        self.current_points_2d=[]
        self.current_points_3d=[]

        self.count = 0
        self.total = 0
        
        self.si=0
        self.p2ss=[]
        self.p3ss=[]

        self.queue = deque(maxlen=seq_len)

        for _ in range(seq_len):
            self.queue.append(np.zeros(n_joints))


    def skeleton_callback(self, pos_msg):
        objs=pos_msg.objects
        # print('\npose: ',len(objs), pos_msg)
        objs = pos_msg.objects
        points_2d=[]
        points_3d=[]
        for obj in objs:
            isk=obj.skeleton_available
            if isk:
                body_format=obj.body_format          #2->38 keypoints
                kps=obj.skeleton_2d.keypoints
                kps3d=obj.skeleton_3d.keypoints

                for id in self.kp_indices:
                    # points_2d.append(kps[id].kp)
                    points_2d.append(kps[id].kp[0])
                    points_2d.append(kps[id].kp[1])

                    points_3d.append(kps3d[id].kp)
                # print(f'\nSkeleton: {isk} {body_format} {len(kps)} {len(kps3d)}')
                break #only one person
        if len(points_2d)==0:
            points_2d=np.zeros(n_joints)

        # print('points_2d=', len(points_2d))
        self.current_points_2d=points_2d
        self.current_points_3d=points_3d

        self.queue.append(points_2d)

        pred=-1
        if len(self.queue) == seq_len:
            # print('time to predict')
            action=np.array(self.queue)
            tx=torch.from_numpy(action).float()
            tx=tx.unsqueeze(0) #.to(device)
            output = model(tx).cpu().detach().numpy()
            po=output.argmax(axis=1)
            pred=po[0] 
            print('pred=', pred)

            self.count = self.count+1
            self.total = self.total + pred
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
            
            # msg = String()
            # msg.data = 'eating' if pred==1 else 'not eating'
            # self.publisher_.publish(msg)

        else:
            print("seq len=", len(self.queue))

    def image_callback(self, image_msg):
        if image_msg is None:
            return
 
        self.si+=1
        self.cv2_image = self.bridge.imgmsg_to_cv2(image_msg,
                                                    desired_encoding='passthrough')  # Preserve original encoding
        # bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        width,height=self.cv2_image.shape[1],self.cv2_image.shape[0]
        p2s=self.current_points_2d
        p3s=self.current_points_3d

        #draw serial number in the image
        cv2.putText(self.cv2_image, str(self.si), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # bgr_img = cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2BGR)
        if view_img:
            cv2.imshow('topic_image',self.cv2_image)
            if cv2.waitKey(1) == 27: 
                self.close() 
                print('UI closed')
 

    def close(self):
        print('Closing')
        if view_img:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    node = ZedInf()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

