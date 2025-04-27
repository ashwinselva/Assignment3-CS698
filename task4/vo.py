#This code took 10 minutes to run in my pc. If ran please do wait for the plot to be plotted

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def load_kitti_calibration(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0_line = [l for l in lines if l.startswith('P0')][0]
    P0_values = np.array(P0_line.strip().split(' ')[1:], dtype=float).reshape(3, 4)
    K = P0_values[:, :3]  
    return K

def load_images_from_folder(folder_path):
    img_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    return img_files

def visual_odometry(img_folder, calib_path):
    K = load_kitti_calibration(calib_path)
    
    img_files = load_images_from_folder(img_folder)
    print(f"Loaded {len(img_files)} images.")

    orb = cv2.ORB_create(5000)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    trajectory = []
    cur_pose = np.eye(4)

    for idx in range(len(img_files) - 1):
        img1 = cv2.imread(img_files[idx], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_files[idx+1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Error loading images {img_files[idx]} or {img_files[idx+1]}")
            continue

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            print(f"Warning: No descriptors found in frame {idx}.")
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            print(f"Warning: Not enough matches in frame {idx}.")
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            print(f"Warning: Essential matrix not found at frame {idx}.")
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()

        cur_pose = cur_pose @ np.linalg.inv(T)

        trajectory.append(cur_pose[:3, 3])

    return np.array(trajectory)

def plot_trajectory(trajectory):
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Estimated Trajectory (Visual Odometry)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    img_folder = './dataset/sequences/00/image_0/'
    calib_path = './dataset/sequences/00/calib.txt'

    trajectory = visual_odometry(img_folder, calib_path)

    print(f"Computed trajectory with {trajectory.shape[0]} poses.")

    plot_trajectory(trajectory)
