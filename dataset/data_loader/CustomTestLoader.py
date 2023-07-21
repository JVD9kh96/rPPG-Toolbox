"""The dataloader for Custom data loader dataset.

"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class CustomTestLoader(BaseLoader):
    """The data loader for an arbitrarily chosen video!"""

    def __init__(self, name, data_path, config_data):
        """Initializes the Custom test data dataloader.
            Args:
                data_path(str): path of a test_data folder:
                -----------------
                     test_data/
                     |   |-- videos/
                     |       |-- vidName1.avi (or mp4, .... (whatever is supported by cv2))
                     |       |-- vidName2.avi (or mp4, .... (whatever is supported by cv2))
                     |       |...
                     |       |-- vidNameN.avi (or mp4, .... (whatever is supported by cv2))
                     |   |-- rPPG/
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        # data_dirs = glob.glob(data_path + os.sep + "subject*")
        # if not data_dirs:
        #     raise ValueError(self.dataset_name + " data paths empty!")
        # dirs = [{"index": re.search(
        #     'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        # return dirs

        data_dirs = glob.glob(data_path + os.sep +  'videos' + os.sep + '*')
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": os.path.split(data_dir)[-1].split('.')[0], 
                  "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Does nothing here!"""
        return data_dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        frames = self.read_video(data_dirs[i]['path'])

        # Read Labels, actually, it generates psuedo labels!
        bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)