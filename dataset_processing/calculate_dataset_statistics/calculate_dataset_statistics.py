import numpy as np
import sys
import os
import argparse
import random
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from skimage import io
from utils_gen import model_utils

def main():
    parser = argparse.ArgumentParser(description="Dataset statistics calculator")
    parser.add_argument(
        "--dataset-path",
        help="Specify the dataset path",
        default="Nothing",
    )
    parser.add_argument(
        "--num-images",
        help="Specify the number of images the statistics will be based on",
        default=5000,
        type=int,
    )
    args = parser.parse_args()
    images_extension = (".jpg", ".jpeg", ".JPEG", ".JPG")

    dataset_path = args.dataset_path
    sample_num_images = args.num_images

    list_of_files_full = collect_desired_sample_images(dataset_path, sample_num_images, images_extension)

    total_average_color = 0
    num_files = len(list_of_files_full)
    counter = 0

    all_images_n_pixels = []
    all_images_means = []
    all_images_std_r = []
    all_images_std_g = []
    all_images_std_b = []

    for image in list_of_files_full:
        counter = counter + 1
        image_current = io.imread(image)[:,:,:]
        print("Processing image {} / {}".format(counter, num_files))

        all_images_std_r.append(np.std(image_current[:, :, 0]))
        all_images_std_g.append(np.std(image_current[:, :, 1]))
        all_images_std_b.append(np.std(image_current[:, :, 2]))

        all_images_means.append([np.mean(image_current[:, :, 0]),
                                np.mean(image_current[:, :, 1]),
                                np.mean(image_current[:, :, 2])])
        all_images_n_pixels.append(image_current[:, :, 0].size)

    r_means_sum_weighted = 0
    g_means_sum_weighted = 0
    b_means_sum_weighted = 0

    total_images_pixes = sum(all_images_n_pixels)

    for counter_c in range(num_files):
        r_means_sum_weighted += all_images_n_pixels[counter_c] * all_images_means[counter_c][0]
        g_means_sum_weighted += all_images_n_pixels[counter_c] * all_images_means[counter_c][1]
        b_means_sum_weighted += all_images_n_pixels[counter_c] * all_images_means[counter_c][2]

    total_mean_r = r_means_sum_weighted / total_images_pixes
    total_mean_g = g_means_sum_weighted / total_images_pixes
    total_mean_b = b_means_sum_weighted / total_images_pixes

    first_formula_term_r = 0
    first_formula_term_g = 0
    first_formula_term_b = 0

    second_formula_term_r = 0
    second_formula_term_g = 0
    second_formula_term_b = 0

    for num_pixels, c_means, stds_r, stds_g, stds_b in zip(all_images_n_pixels, all_images_means,
                                   all_images_std_r, all_images_std_g, all_images_std_b):
        second_formula_term_r += num_pixels * ((c_means[0] - total_mean_r)**2)
        second_formula_term_g += num_pixels * ((c_means[1] - total_mean_g)**2)
        second_formula_term_b += num_pixels * ((c_means[2] - total_mean_b)**2)

        first_formula_term_r += num_pixels * (stds_r**2)
        first_formula_term_g += num_pixels * (stds_g**2)
        first_formula_term_b += num_pixels * (stds_b**2)

    final_std_r = np.sqrt((first_formula_term_r + second_formula_term_r) / total_images_pixes)
    final_std_g = np.sqrt((first_formula_term_g + second_formula_term_g) / total_images_pixes)
    final_std_b = np.sqrt((first_formula_term_b + second_formula_term_b) / total_images_pixes)

    print("Final average color: ", [total_mean_r, total_mean_g, total_mean_b])
    print("Final St. Dev. for all color channels: ", [final_std_r, final_std_g, final_std_b])


def collect_desired_sample_images(dataset_path, num_images_in_sample, images_extensions):
    all_files_full = []

    print("Gathering sample images...")

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in [f for f in filenames if f.endswith(images_extensions)]:
            all_files_full.append(os.path.join(dirpath, filename))

    random.shuffle(all_files_full)

    if num_images_in_sample <= len(all_files_full):
        sample_files_full = all_files_full[0:num_images_in_sample]
    else:
        sample_files_full = all_files_full[:]

    print("Gathered " + str(len(sample_files_full)) + " images for the sample")

    return sample_files_full

if __name__ == "__main__":
    main()