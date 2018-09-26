"""
Downloads and processes drone footage from https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx
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

OUTPUT_DIR = "kaust_uav123_data"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

HOST_NAME = "ftps.kaust.edu.sa"

DATASET_NAME = "Dataset_UAV123_10fps.zip"
DATA_SEQ_FOLDER = os.path.join("UAV123_10fps", "data_seq", "UAV123_10fps")

DESIRED_IM_SZ = (128, 160)

# Leeches are data samples that were hand-determined to be bad training examples for various reasons.
# Reasons for data being a leech include video artifacts, rapid unpredictable movement, too low fps to
# pick up on motion patterns, etc.
LEECHES = {"bird1", "car1_s", "car2_s", "car3_s", "car4_s", "person1_s", "person2_s", "person3_s", 
           "uav1", "uav2", "uav3", "uav4", "uav5", "uav6", "uav7" ,"uav8" # Too many video artifacts to be useful
# potentially "car17", "car18", "person10", "person19", "person20" # Too much unpredictable camera movement to learn
}

# Download the data directly from the FTP server.
def download_data():
    # FTP Configuration found here:
    # https://ivul.kaust.edu.sa/Documents/Data/FTP%20Server%20User%20Guide.pdfS
    ftp = ftplib.FTP_TLS()
    ftp.connect(HOST_NAME)
    ftp.login() # Anonymous login, only accepted on TLS-enabled FTP client.
    ftp.set_pasv(True) # Client must be in Passive mode to retrieve 
    ftp.cwd("UAV")

    output_file_name = os.path.join(OUTPUT_DIR, DATASET_NAME)
    with open(output_file_name, "w+") as output_file:
        ftp.retrbinary("RETR " + DATASET_NAME, output_file.write, blocksize=1024)

    ftp.quit()

# Unzip the data
def extract_data():
    data_zip_name = os.path.join(OUTPUT_DIR, DATASET_NAME)
    match_folders = os.path.join(DATA_SEQ_FOLDER, "*")
    command = " ".join([
        "unzip -qq", data_zip_name, match_folders, "-d", OUTPUT_DIR
        ])

    error = os.system(command)
    if error:
        raise Exception(error)

    os.system("rm " + data_zip_name)

def remove_leeches():
    _, folders, _ = os.walk(os.path.join(OUTPUT_DIR, DATA_SEQ_FOLDER)).next()
    for folder in folders:
        if folder in LEECHES:
            os.system("rm -r " + os.path.join(OUTPUT_DIR, DATA_SEQ_FOLDER, folder))

# Walk through the downloaded data directories and create hickle files
# containing the numpy video array for each video (sized appropriately).
def process_data():
    _, folders, _ = os.walk(os.path.join(OUTPUT_DIR, DATA_SEQ_FOLDER)).next()
    for folder in folders:
        data_root_folder = os.path.join(OUTPUT_DIR, DATA_SEQ_FOLDER, folder)
        _, _, image_files = os.walk(data_root_folder).next()

        video_array = get_video_array([os.path.join(data_root_folder, img) for img in image_files])
        hkl.dump(video_array, os.path.join(data_root_folder, folder + ".hkl"))

# Create the numpy video array corresponding to the given folder of image files.
def get_video_array(image_files):
    X = np.zeros((len(image_files),) + DESIRED_IM_SZ + (3,), np.uint8)
    for i, image_file in enumerate(image_files):
        image = imageio.imread(image_file)
        X[i] = resize(image, DESIRED_IM_SZ)
    return X

def resize(image, desired_sz):
    # Like with the kitti data, we'll just scale so that the image height is the desired size
    # then we'll crop along the width.
    scale_factor = float(desired_sz[0])/image.shape[0]
    resized = imresize(image, 
        (int(scale_factor * image.shape[0]), int(scale_factor * image.shape[1])))
    d = int((resized.shape[1] - desired_sz[1])/2)
    return resized[:, d:d+desired_sz[1]]

if __name__ == "__main__":
    download_data()
    extract_data()
    remove_leeches()
    process_data()