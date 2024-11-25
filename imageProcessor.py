from PIL import Image
import numpy as np
from rl.core import Processor


IMG_SHAPE=(84,84)

class ImageProcessor(Processor):
    def process_observation(self, observation):
        img=Image.fromarray(observation)
        img=img.resize(IMG_SHAPE)
        img=img.convert("L")
        img=np.array(img)
        return img.astype('uint8')
    
    def process_state_batch(self, batch):
        processed_batch=batch.astype('float32')/255.0
        return processed_batch
    