# Alex Shrestha
# FILE: image_stitch.py 

import cv2
import sys
import numpy as np

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    """
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    """
    best_H = None

    # to be completed ...

    return best_H

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    """
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    """

    sift = cv2.SIFT_create()
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    gray1 = cv.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    
    gray2 = cv.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []

    for f1 in range(desc1.shape[0]):
        ssd = []
        for f2 in range(desc2.shape[0]):
            diff = desc1[f1] - desc2[f2]
            ssd.append(np.sqrt(np.dot(diff, diff)))

        ssd.sort()
        
        if ssd[0]/ssd[1] <= ratio_robustness:
            x, y = kp1[f1].pt
            list_pairs_matched_keypoints.append([x, y])

    return list_pairs_matched_keypoints

def ex_warp_blend_crop_image(img_1,H_1,img_2):
    """
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    """

    img_panorama = None
    # TODO

    # ===== blend images: average blending
    # TODO

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # TODO

    return img_panorama

def stitch_images(img_1, img_2):
    """
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    """

    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)


    return img_panorama

if __name__ == "__main__":
    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))

