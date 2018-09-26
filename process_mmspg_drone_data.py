"""
Downloads and processes drone footage from https://mmspg.epfl.ch/mini-drone
"""

import ftplib
import hickle as hkl
import imageio
import numpy as np
import os
import warnings

# Check scipy version for deprecated imread
from scipy import __version__ as scipy_version
if scipy_version >= '1.2.0':
    from skimage.transform import resize as imresize
else:
    from scipy.misc import imresize

OUTPUT_DIR = "mmspg_drone_data"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

HOST_NAME = "tremplin.epfl.ch"
USER_NAME = "minidrone@grebvm2.epfl.ch"
PASSWORD  = "IH)cJ9c81*1H74kv"

DESIRED_IM_SZ = (128, 160)
DESIRED_FPS = 10

# Download the data directly from the FTP server.
def download_data():
    ftp = ftplib.FTP()
    ftp.connect(HOST_NAME)
    ftp.login(USER_NAME, PASSWORD)

    # The data on the server is split into training and testing data.
    split_names = ftp.nlst()
    for split in split_names:
        ftp.cwd(split)

        videos = filter(lambda x: x.endswith(".mp4"), ftp.nlst())
        for video_name in videos:
            video_output_file = os.path.join(OUTPUT_DIR, split, video_name[:-4], video_name)
            # video_name[:-4] just removes the ".mp4" from the end.

            if not os.path.exists(os.path.dirname(video_output_file)):
                os.makedirs(os.path.dirname(video_output_file))

            video_file = open(video_output_file, "w+")

            ftp.retrbinary("RETR " + video_name, video_file.write, blocksize=1024)

            video_file.close()
        ftp.cwd("/")
    ftp.quit()


# Walk through the downloaded data directories and create hickle files
# containing the numpy video array for each video (sized appropriately).
def process_data():
    _, splits, _ = os.walk(OUTPUT_DIR).next()
    for split in splits:
        split_dir = os.path.join(OUTPUT_DIR, split)
        _, dirs, _ = os.walk(split_dir).next()
        
        for dir in dirs:
            
            video_dir = os.path.join(split_dir, dir)
            _, _, files = os.walk(video_dir).next()
            if not len(files):
                warnings.warn("No video file found in " + video_dir)
            
            if len(files) > 1:
                # just assume we're done parsing it
                continue
            
            video_file = os.path.join(video_dir, files[0])

            video_array = get_video_array(video_file)
            save_video_array_as_images(video_file, video_array)

            hkl.dump(video_array, video_file[:-4] + ".hkl")

# Given the name of a video_file to read, return the numpy array of
# images corresponding to this video.
def get_video_array(video_file):
    video_reader = imageio.get_reader(video_file, format="ffmpeg")
    meta_data = video_reader.get_meta_data()

    original_fps = meta_data["fps"]
    step_rate = original_fps/float(DESIRED_FPS)
    # Note: will increasing the frame_index by a fractional step_rate 
    # cause the speed of the output video to fluctuate too much?
    if step_rate < 1:
        warnings.warn("Desired FPS is greater than original FPS in video " + video_file + \
                      ". This will cause frame repetition.")
    
    images = []

    input_frame_index = 0
    while input_frame_index < video_reader.get_length() - 1:
        image = video_reader.get_data(int(input_frame_index))
        # We just use the vanilla imresize here (without worrying about aspect ratio)
        # because it looks similar enough to the original video that it won't cause any
        # problems in the prednet training.

        # NOTE: this *might* cause problems in prediction if the network somehow encodes
        # aspect ratio information, but this is unlikely (and untested).
        resized_image = imresize(image, DESIRED_IM_SZ)
        images.append(resized_image)

        input_frame_index += step_rate
    
    video_array = np.zeros((len(images),) + DESIRED_IM_SZ + (3,), np.uint8)
    for i, image in enumerate(images):
        video_array[i] = image

    return video_array

# Save a numpy video array as a bunch of image files.
def save_video_array_as_images(video_file, video_array):
    video_name = video_file[:-4] if video_file.endswith(".mp4") else video_file

    for i, image in enumerate(video_array):
        image_name = video_name + "_" + str(i) + ".png"
        imageio.imwrite(image_name, image)

if __name__ == "__main__":
    download_data()
    process_data()