from abc import ABC, abstractmethod
from collections import deque

from numpy import ndarray
import cv2

class Flow(ABC):
    def __init__(self, width:int, height:int, *args, **kwargs):
        self.width = width
        self.height = height
    
    @abstractmethod
    def calc(self, prev, frame):
        pass

class NvidiaFlow(Flow):
    def __init__(self, width:int, height:int, perfPreset=5, gpuId=0, **kwargs):
        super().__init__(width, height)

        assert perfPreset in {5, 10, 20}, f"Performance preset must be 5 (slow), 10 (medium), or 20 (fast). Provided: {perfPreset}"
        self.perfPreset = perfPreset
        self._flow = cv2.cuda_NvidiaOpticalFlow_1_0.create(width, height, perfPreset=perfPreset, gpuId=gpuId, **kwargs)
        self.gridSize = self._flow.getGridSize()
    
    def calc(self, prev, frame):
        flow = self._flow.calc(prev, frame, None)

        return self._flow.upSampler(flow[0], self.width, self.height, self.gridSize, None)


class FarnebackFlow(Flow):
    def __init__(self, width:int, height:int, pyr_scale=0.5, levels=3,
                    winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0, **kwargs):
        super().__init__(width, height)
        
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
    
    def calc(self, prev, frame):
        return cv2.calcOpticalFlowFarneback(prev, frame, None, self.pyr_scale,
                                            self.levels, self.winsize, self.iterations,
                                            self.poly_n, self.poly_sigma, self.flags)


# iterator classes
class FlowIterator(ABC):

    modes = {
        0: cv2.COLOR_BGR2GRAY,
        1: cv2.COLOR_RGB2GRAY,
        2: None,
        3: cv2.COLOR_YUV2GRAY_YUYV,
    }

    def __init__(self, video, *args, frame_distance:int=1, **kwargs):
        if not video.isOpened():
            raise Exception("Could not open video")
        
        self.video = video

        mode = video.get(cv2.CAP_PROP_MODE)
        self.cvtColor = FlowIterator.modes[mode]

        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frame_distance = frame_distance

        assert self.frame_distance > 0, f"frame_distance must be a positive integer: ({self.frame_distance} !> 0)"
        assert self.frame_distance < self.video.get(cv2.CAP_PROP_FRAME_COUNT), f"frame_distance must be less than the number of frames in the video ({self.frame_distance} !< {self.video.get(cv2.CAP_PROP_FRAME_COUNT)})"
        
        self.queue = deque(maxlen=frame_distance)

        self.flow = self.initialize_flow(video, *args, **kwargs)
        
        # frame index starts at 0
        # self.index = 0
        for _ in range(frame_distance):
            ret, frame = self.video.read()
            if not ret:
                raise Exception("Error reading video")
            # self.index += 1
            self.queue.append(cv2.cvtColor(frame, self.cvtColor))

    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) - self.frame_distance

    def __iter__(self):
        return self
    
    def release(self):
        self.video.release()
    
    @abstractmethod
    def initialize_flow(self, video, *args, **kwargs):
        pass

    def __next__(self):
        ret, frame = self.video.read()

        if not ret:
            self.video.release()
            raise StopIteration

        gray = cv2.cvtColor(frame, self.cvtColor)
        
        flow = self.flow.calc(self.queue[0], gray)
        
        self.queue.append(gray)

        return flow, frame

class NvidiaFlowIterator(FlowIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize_flow(self, video, *args, **kwargs):
        return NvidiaFlow(self.width, self.height, *args, **kwargs)

    def __next__(self):
        ret, frame = self.video.read()

        if not ret:
            self.video.release()
            self.flow._flow.collectGarbage()
            raise StopIteration

        gray = cv2.cvtColor(frame, self.cvtColor)
        
        flow = self.flow.calc(self.queue[0], gray)
        
        self.queue.append(gray)

        return flow, frame

class FarnebackFlowIterator(FlowIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize_flow(self, video, *args, **kwargs):
        return FarnebackFlow(self.width, self.height, *args, **kwargs)

# global methods

def to_polar(flow) -> (ndarray, ndarray):
    return cv2.cartToPolar(flow[...,0], flow[...,1])