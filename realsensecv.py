import pyrealsense2 as rs
import numpy as np

# 1920 x 1080 w/o depth image

# class RealsenseCapture:
#     def __init__(self):
#         self.WIDTH = 1920
#         self.HEGIHT = 1080
#         self.FPS = 30
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
#         # self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)

#     def start(self):
#         self.pipeline = rs.pipeline()
#         self.pipeline.start(self.config)
#         print('pipline start')

#     def read(self, is_array=True):
#         ret = True
#         frames = self.pipeline.wait_for_frames()
#         self.color_frame = frames.get_color_frame()  # RGB
#         # self.depth_frame = frames.get_depth_frame()  # Depth
#         if not self.color_frame:
#             ret = False
#             return ret, (None, None)
#         elif is_array:
#             color_image = np.array(self.color_frame.get_data())
#             # depth_image = np.array(self.depth_frame.get_data())
#             return ret, (color_image)
#         else:
#             return ret, (self.color_frame)

#     def release(self):
#         self.pipeline.stop()

# 1280 x 720 w/ depth image

class RealsenseCapture:
    def __init__(self):
        self.WIDTH = 1280
        self.HEGIHT = 720
        self.FPS = 30
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)

    def start(self):
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)
        print('pipline start')

    def read(self, is_array=True):
        ret = True
        frames = self.pipeline.wait_for_frames()
        self.color_frame = frames.get_color_frame()  # RGB
        self.depth_frame = frames.get_depth_frame()  # Depth
        if not self.color_frame or not self.depth_frame:
            ret = False
            return ret, (None, None)
        elif is_array:
            color_image = np.array(self.color_frame.get_data())
            depth_image = np.array(self.depth_frame.get_data())
            return ret, (color_image, depth_image)
        else:
            return ret, (self.color_frame, self.depth_frame)

    def release(self):
        self.pipeline.stop()