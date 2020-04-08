import os
import skimage.io
import skimage.color
import numpy as np

class CDataset(object):
    def __init__(self):
        self._image_ids = []
        self.image_info = []
        self.batch = 0
        
    def load_img_from_dir(self, data_dir):
        for filename in os.listdir(data_dir):
            self.add_image(
                    image_id=filename,
                    path=os.path.join(data_dir, filename))

    def add_image(self, image_id, path, **kwargs):
        """
        id: filename
        path: file path
        """
        image_info = {
                "id": image_id,
                "path": path,
                }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # Check that it is an rgb image
        if image.ndim != 3:
            return -1
        rgb = np.copy(image)
        # Convert to grayscale
        image = skimage.color.rgb2gray(image)
        # Remove any apha channel
        if image.shape[-1] == 4:
            image = image[..., :3]
        h,w = image.shape[0:2]
        image = np.reshape(image, (1,h,w,1))
        return rgb, image.astype(np.float32)

    @property
    def image_ids(self):
        return self._image_ids

    @image_ids.setter
    def image_ids(self, value):
        self._image_ids = value

    def prepare(self, shuffle=True):
        self.b = 0
        # Two images per batch
        self.num_batch = len(self.image_info)//2
        self.num_images = self.num_batch*2
        self.image_ids = np.arange(self.num_images)
        # Shuffle
        if shuffle:
            self.image_ids = np.random.shuffle(self.image_ids)

    def get_next(self):
        image_id1 = self.b*2
        image_id2 = image_id1+1
        rgb1, grey1 = self.load_image(image_id1)
        rgb2, grey2 = self.load_image(image_id2)

        self.b += 1
        if self.b == self.num_batch:
            self.b = 0

        return rgb1, grey1, rgb2, grey2
