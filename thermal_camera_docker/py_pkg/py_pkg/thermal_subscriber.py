#!/usr/bin/env python3
from asyncio import futures
from distutils.archive_util import make_archive
import imp
from inspect import Parameter
from cv2 import WARP_INVERSE_MAP, cvtColor, waitKey
from numpy import array, interp
import numpy
import rclpy
import numpy as np
import cv2 as cv
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from lifecycle_msgs.msg import State
import math

class ThermalSubscriberNode(Node): 
    def __init__(self):
        super().__init__("thermal_subscriber")
        self.subscriber = self.create_subscription(Image, "thermal_image", self.dataReceivedCallback, qos_profile=10)
        self.frame = [0 for i in range(0, 576)]
        self.get_logger().info("Node started")
        self.iterator = 0
        self.buf= []
        self._markerPublisher = self.create_publisher(Marker, "tc_goal_angle", 10)
        self.imagePublisher = self.create_publisher(Image, "tc_image_devel", 10)
        self.posePublisher = self.create_publisher(PoseStamped, "goal_pose", 10)
        self.cli = self.create_client(GetState, 'bt_navigator/get_state')
        self.navStateRequest = GetState.Request()
        self.state = State()
        self.goalPose = PoseStamped()
        self.images = []
        self.targetAngle = 0.0
        self.marker = Marker()
        self.lastAngle = 0
        self.imageToPublish = []
        self.stitcher = cv.Stitcher_create()
        self.lastCenterPixel = 0
        self.targetTempMin = 30
        self.targetTempMax = 37
        self.goToHottest = False
        self.certain = 0
        self.timer = self.create_timer(0.5, self.parseParams)
        self.declare_parameter('target_min_temp', 34)
        self.declare_parameter('target_max_temp', 37)
        self.declare_parameter('go_to_hottest_point', False)
    # def publishImageCallback(self):
    #     msg = Image()
    #     msg.header.frame_id = "map"
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.width = 16*3
    #     msg.height = 12
    #     msg.encoding = "mono8"
    #     msg.is_bigendian = False
    #     msg.step = 16*3
    #     msg.data = []
    #     for i in range(16*3*12):
    #             msg.data.append(self.frame[i])
        
    #     self._imagePublisher.publish(msg)
    def parseParams(self):
        self.targetTempMin = self.get_parameter('target_min_temp').get_parameter_value().integer_value
        self.targetTempMax = self.get_parameter('target_max_temp').get_parameter_value().integer_value
        self.goToHottest = self.get_parameter('go_to_hottest_point').get_parameter_value().bool_value
        self.get_logger().info("target min temp: {}     target max temp: {}     go to hottest: {}".format(self.targetTempMin, self.targetTempMax, self.goToHottest))
    def sendGetStateRequest(self):
        future = self.cli.call_async(self.navStateRequest)
        rclpy.spin_until_future_complete(self, future)
        if(future.result() is not None):
            self.get_logger().info(str(future.result()))

    def publishArrow(self, angle):
        self.marker.header.frame_id = "base_link"
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.ns = "basic_shapes"
        self.marker.id = 0
        self.marker.type = Marker.ARROW
        self.marker.action = Marker.ADD
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0
        self.marker.scale.x = 1.0
        self.marker.scale.y = 0.05
        self.marker.scale.z = 0.05
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0
        
        self.marker.pose.orientation.x = math.sin(3.1415 * (angle-180)/360)
        self.marker.pose.orientation.y = math.cos(3.1415 * (angle-180)/360)
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 0.0
        
        self.goalPose.header.frame_id = "base_link"
        self.goalPose.header.stamp = self.get_clock().now().to_msg()
        self.goalPose.pose.position.x = 0.0
        self.goalPose.pose.position.y = 0.0
        self.goalPose.pose.position.z = 0.0
        self.goalPose.pose.orientation.x = math.sin(3.1415 * (angle-180)/360)
        self.goalPose.pose.orientation.y = math.cos(3.1415 * (angle-180)/360)
        self.goalPose.pose.orientation.z = 0.0
        self.goalPose.pose.orientation.w = 0.0
        
        self.sendGetStateRequest()
        if(-5 <= angle <= 5):
            self.goalPose.pose.position.x = 0.5
        
        self.posePublisher.publish(self.goalPose)
        self._markerPublisher.publish(self.marker)

    def publishImage(self, image):
        bridge = CvBridge()
        msg = bridge.cv2_to_imgmsg(image)
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.imagePublisher.publish(msg)

    def dataReceivedCallback(self, msg):
        #OpenCV Bridge
        bridge = CvBridge()
        #normalizing input image
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = np.zeros_like(raw_image)
        cv_image = cv.normalize(raw_image, cv_image, 0, 65535, cv.NORM_MINMAX)

        #converting from 16bit mono to 8 bit 
        img8bit = (cv_image/256).astype("uint8")

        #creating color image that can display colored contours and markers
        color_img = cvtColor(img8bit, cv.COLOR_GRAY2RGB)

        #resizing image 3 times to increase clarity
        color_img = cv.resize(color_img, (16*9, 12*3), interpolation=cv.INTER_LINEAR)

        if(self.goToHottest == True):
            #creating a mask image that contains the hotest pixels
            ret, bin_img = cv.threshold(img8bit, 165, 255, cv.THRESH_BINARY)
            bin_img = cv.resize(bin_img, (16*9, 12*3), interpolation=cv.INTER_LINEAR)
            contours, hier = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            index = 0
            warmest = 0 

            #creating a mask to find group of the hotest pixels
            mask = np.zeros_like(bin_img)
            for i in range(len(contours)):
                cv.drawContours(mask, contours, i, (255,255,255), thickness=cv.FILLED)
                temp = cv.mean(color_img, mask=mask)[0]
                if(temp>warmest and len(contours[i]>6)):
                    warmest = temp
                    index = i

            #bounding rect helps finding center of the contour
            x,y,w,h = cv.boundingRect(contours[index])
            centerPixel = int(x+(w)/2)
        else:
            index = 0
            ret, bin_img = cv.threshold(raw_image, self.targetTempMin*100, self.targetTempMin*100, cv.THRESH_BINARY)
            bin_img = cv.resize(bin_img, (16*9, 12*3), interpolation=cv.INTER_LINEAR)
            bin_img = (bin_img/256).astype("uint8")
            contours, hier = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.normalize(bin_img, bin_img, 0, 255, cv.NORM_MINMAX)
            largestArea = 0
            for i in range(len(contours)):
                area = cv.contourArea(contours[i])
                if(area>largestArea):
                    largestArea = area
                    index = i
            
            self.get_logger().info("LEN: {}         INDEX: {}".format(len(contours), index))
            if(len(contours)>0):
                x,y,w,h = cv.boundingRect(contours[index])
                centerPixel = int(x+(w)/2)
                self.lastCenterPixel = centerPixel
            else:
                centerPixel = self.lastCenterPixel
        #converting pixel position to angle
        hottestAngle = -165 + centerPixel*6.875 /3


        #average the goal point
        self.buf.append(hottestAngle)
        if(len(self.buf)>=10):
            self.buf.pop(0)
        mean = 0
        for vals in self.buf:
            mean += vals
        self.targetAngle = mean/10
        if(self.iterator<10):
            self.iterator +=1
        if(self.iterator>=10):
            self.iterator = 0



        #calculating markers position based on averaged angle
        redMarker = int((165 + self.targetAngle)*3/6.875)

        self.lastAngle = hottestAngle
        cv.drawMarker(color_img, (centerPixel, 6*3), (255,0,0))
        cv.drawMarker(color_img, (redMarker, 6*3), (0,0,255))

        #self.get_logger().info("Target angle: {:.2f}      Hottest angle: {:.2f}".format(self.targetAngle, hottestAngle))

        cv.drawContours(color_img, contours, index, (0,255,0), 1)

        #cv.imshow("Thermal Camera Feed", color_img)
        #waitKey(1)
        self.publishImage(color_img)
        self.publishArrow(self.targetAngle)
        # temps = []
        # for i in range(0,msg.width * msg.height * 2, 2):
        #     temps.append(float(msg.data[i] | msg.data[i+1]<<8)/100)
            
        # self.get_logger().info(str(temps) + "\n" + str(len(temps)))

    # def processData(self, data:Image):
    #     imageID = data[0]
    #     for i in range(16*12):
    #         self.frame[int(i/16)*32 + i + (16*imageID)] = data[i+1]
    #     if(imageID == 0): 
    #         self.publishImageCallback()

def main(args=None):
    rclpy.init(args=args)
    node = ThermalSubscriberNode() 
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
