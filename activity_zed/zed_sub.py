import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from zed_interfaces.msg import ObjectsStamped
import cv2
from cv_bridge import CvBridge
import os
import rclpy
from rclpy.node import Node
import imageio 
import datetime 

class ZedSub(Node):

    def __init__(self):
        super().__init__('zed_sub')
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

        now = datetime.datetime.now()
        time_str=now.strftime("%m_%d_%Y_%H_%M")
        self.savedir="/home/ns/ros2_ws/videos/"+time_str+"/"
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        
        video_path=self.savedir+'zed.mp4'
        print('saving to: ',video_path)
        self.video_writer = imageio.get_writer(video_path, fps=5)
        self.si=0
        self.p2ss=[]
        self.p3ss=[]


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
                    points_2d.append(kps[id].kp)
                    points_3d.append(kps3d[id].kp)
                # print(f'\nSkeleton: {isk} {body_format} {len(kps)} {len(kps3d)}')
                break #only one person

        self.current_points_2d=points_2d
        self.current_points_3d=points_3d
        

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

        p2s_str=''
        p3s_str=''
        for i in range(len(p2s)):
            p2=p2s[i]
            p3=p3s[i]
            p2s_str+=f'{int(p2[0])},{int(p2[1])},'
            p3s_str+=f'{p3[0]},{p3[1]},{p3[2]},' 
 
            # x,y=int(width*p2[0]),int(height*p2[1])
            x,y=int(p2[0]),int(p2[1])
            if x<0 or y<0:
                continue

            original_size = (720, 1080)  # Original image size (width, height)
            new_size = (height, width)

            scale_x= new_size[0] /original_size[0]
            scale_y= new_size[1] /original_size[1]

            x=int(x*scale_x)
            y=int(y*scale_y)

            # print(width, height, x,y , p3[0], p3[1], p3[2])
            # cv2.circle(self.cv2_image, (x, y), 5, (255, 0, 0), -1)  

        # for i in range(0,len(p2s),2):
        #     x1,y1=int(p2s[i][0]),int(p2s[i][1])
        #     x2,y2=int(p2s[i+1][0]),int(p2s[i+1][1])
        #     cv2.line(self.cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.p2ss.append(p2s_str)
        self.p3ss.append(p3s_str) 

        #draw serial number in the image
        cv2.putText(self.cv2_image, str(self.si), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        bgr_img = cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2BGR)
        self.video_writer.append_data(bgr_img)
        cv2.imshow('topic_image',self.cv2_image)
        if cv2.waitKey(1) == 27: 
            self.close() 
            print('UI closed')
 

    def close(self):
        print('Closing')
        with open(self.savedir+'p2s.txt', 'w') as f:
            for item in self.p2ss:
                f.write("%s\n" % item)
        with open(self.savedir+'p3s.txt', 'w') as f:
            for item in self.p3ss:
                f.write("%s\n" % item)

        self.video_writer.close()
        print('Video and 2D,3D keypoints saved')

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    node = ZedSub()

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

