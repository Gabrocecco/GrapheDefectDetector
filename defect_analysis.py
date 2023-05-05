try:
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm
    from skimage.feature import hog
    from scipy.fftpack import fft2, fftshift
    from mahotas.features import haralick
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
except Exception as e:
    print(f"Some module are missing for {__file__}: {e}\n")


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

def contrast_enhancement(image: Path | np.ndarray, dest_path: Path | None = None):
    # Load the image
    if isinstance(image, Path):
        img = cv2.imread(str(image), 0)
    else:
        img = image.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equ = cv2.equalizeHist(img)

    if dest_path is not None:
        cv2.imwrite(str(dest_path), equ)
    else:
        return equ


        def crop_melanoma(
            image: np.ndarray,
            mask: np.ndarray,
        ) -> np.ndarray:
            non_empty_columns = np.where(mask.max(axis=0) > 0)[0]
            non_empty_rows = np.where(mask.max(axis=1) > 0)[0]
            cropBox = (
                min(non_empty_rows),
                max(non_empty_rows),
                min(non_empty_columns),
                max(non_empty_columns),
            )

            if len(image.shape) == 3:
                crop = image[cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :]
            elif len(image.shape) == 2:
                crop = image[cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1]
            else:
                raise Exception(f"Wrong image shape for: {image}")

            return crop

class Test:
    
    def test_melanoma_mask(img: Path, mask: Path, dpath: Path, gray: bool = False):
        img = cv2.imread(str(img))
        mask = cv2.imread(str(mask))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Convert image to grayscale
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract the melanoma part of the image using the mask
        melanoma = cv2.bitwise_and(img, img, mask=mask)

        cv2.imwrite(str(dpath), melanoma)

    def test_same_shape_img_mask(images_path: Path, masks_path: Path):
        images = [
            f for f in images_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images = sorted(images, key=lambda x: str(x).lower())

        masks = [
            f for f in masks_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        masks = sorted(masks, key=lambda x: str(x).lower())

        count = 0

        pbar = tqdm(total=len(images))
        for image, mask in zip(images, masks):
            img = cv2.imread(str(image), 0)
            msk = cv2.imread(str(mask), 0)

            if not img.shape == msk.shape:
                count += 1

            pbar.update(1)

        pbar.close()
        print(f"Number of shape mismatch: {count} on {len(images)} total samples\n")


    def load_image(image_path: Path | str) -> np.ndarray:
        if isinstance(image_path, Path):
            return cv2.imread(str(image_path))
        elif isinstance(image_path, str):
            return cv2.imread(image_path)
        else:
            raise Exception(f"Wrong type for {image_path}")


    def load_mask(mask_path: Path | str) -> np.ndarray:
        if isinstance(mask_path, Path):
            mask = cv2.imread(str(mask_path))
            return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif isinstance(mask_path, str):
            mask = cv2.imread(mask_path)
            return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception(f"Wrong type for {mask_path}")


    def extract_texture_features(
        image: Path | np.ndarray, mask: Path | np.ndarray = None
    ) -> dict:
        # Load image
        if isinstance(image, Path):
             img = cv2.imread(str(image))
        # else:
        #     img = image.copy()

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()

            # gray = crop_melanoma(gray, mask_img)

        # Compute GLCM matrix
        glcm = graycomatrix(
            gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
        )

        # Compute texture features
        contrast = graycoprops(glcm, "contrast")[0][0]
        homogeneity = graycoprops(glcm, "homogeneity")[0][0]
        energy = graycoprops(glcm, "energy")[0][0]
        correlation = graycoprops(glcm, "correlation")[0][0]

        # Compute LBP features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius)
        (hist, _) = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
        )
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7

        texture_features = {
            "GLCM_contrast": contrast,
            "GLCM_homogeneity": homogeneity,
            "GLCM_energy": energy,
            "GLCM_correlation": correlation,
            "LBP_histogram": hist.tolist(),
        }

        return texture_features


    # def extract_color_features(
    #     image: Path | np.ndarray, mask: Path | np.ndarray = None, hue: bool = False
    # ) -> dict:
    #     # Load image
    #     if isinstance(image, Path):
    #         img = cv2.imread(str(image))
    #     else:
    #         img = image.copy()

    #     if mask is not None:
    #         if isinstance(mask, Path):
    #             mask_img = cv2.imread(str(mask))
    #             mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    #         else:
    #             mask_img = mask.copy()

    #     if hue:
    #         # Convert image to HSV color space
    #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #         if mask is not None:
    #             melanoma = cv2.bitwise_and(hsv, hsv, mask=mask_img)
    #             # Compute mean and standard deviation of hue, saturation, and value channels for the melanoma part
    #             h_mean, h_std = cv2.meanStdDev(melanoma[:, :, 0], mask=mask)
    #             s_mean, s_std = cv2.meanStdDev(melanoma[:, :, 1], mask=mask)
    #             v_mean, v_std = cv2.meanStdDev(melanoma[:, :, 2], mask=mask)
    #         else:
    #             # Compute mean and standard deviation of hue, saturation, and value channels
    #             h_mean, h_std = cv2.meanStdDev(hsv[:, :, 0])
    #             s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
    #             v_mean, v_std = cv2.meanStdDev(hsv[:, :, 2])

    #         # Create dictionary of color features
    #         color_features = {
    #             "hue_mean": h_mean[0][0],
    #             "hue_std": h_std[0][0],
    #             "saturation_mean": s_mean[0][0],
    #             "saturation_std": s_std[0][0],
    #             "value_mean": v_mean[0][0],
    #             "value_std": v_std[0][0],
    #         }

    #     else:
    #         if mask is not None:
    #             melanoma = cv2.bitwise_and(img, img, mask=mask_img)
    #             # Compute mean and standard deviation of hue, saturation, and value channels for the melanoma part
    #             r_mean, r_std = cv2.meanStdDev(melanoma[:, :, 0], mask=mask)
    #             g_mean, g_std = cv2.meanStdDev(melanoma[:, :, 1], mask=mask)
    #             b_mean, b_std = cv2.meanStdDev(melanoma[:, :, 2], mask=mask)
    #         else:
    #             # Compute mean and standard deviation of each color channel
    #             r_mean, r_std = cv2.meanStdDev(img[:, :, 2])
    #             g_mean, g_std = cv2.meanStdDev(img[:, :, 1])
    #             b_mean, b_std = cv2.meanStdDev(img[:, :, 0])

    #         # Create dictionary of color features
    #         color_features = {
    #             "red_mean": r_mean[0][0],
    #             "green_mean": g_mean[0][0],
    #             "blue_mean": b_mean[0][0],
    #             "red_std": r_std[0][0],
    #             "green_std": g_std[0][0],
    #             "blue_std": b_std[0][0],
    #         }

    #     return color_features


    def extract_shape_features(
        image: Path | np.ndarray,
        mask: Path | np.ndarray = None,
        dest_path: Path | None = None,
    ) -> dict:
        # Load image as grayscale
        if isinstance(image, Path):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        # else:
        #     img = image.copy()    #da errore anche se non ci entra 
        # img = contrast_enhancement(img)  # per migliorare la detection dei contorni 

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()

        #     #img = crop_melanoma(img, mask_img)

        # Threshold image to create binary mask
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

        # Filter contours based on area
        contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

        # for contour in contours:
        #     cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
        #     plt.figure()
        #     plt.imshow(img)

        # Get largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter**2)
        solidity = (
            cv2.contourArea(largest_contour)
            / cv2.convexHull(largest_contour, returnPoints=False).size
        )

        compactness = perimeter**2 / area

        _, (diam_x, diam_y), _ = cv2.minAreaRect(largest_contour)
        feret_diameter = max(diam_x, diam_y)

        if diam_x < diam_y:
            diam_x, diam_y = diam_y, diam_x

        eccentricity = np.sqrt(1 - (diam_y / diam_x) ** 2)
        
        if dest_path is not None:
            # Draw largest contour on input image
            img_with_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_with_contour, [largest_contour], -1, (0, 0, 255), 1)
            cv2.imwrite(str(dest_path), img_with_contour)

        # Create dictionary of shape features
        num_pixels = img.shape[0] * img.shape[1]
        shape_features = {
            "area": area / num_pixels,
            # "area": area,   #modifica per avere l'area assoluta 
            "perimeter": perimeter / num_pixels,
            "circularity": circularity,
            "solidity": solidity,
            "compactness": compactness / num_pixels,
            "feret_diameter": feret_diameter / np.sqrt(num_pixels),
            "eccentricity": eccentricity,
        }

        return shape_features


    def extract_edge_features(
        image: Path | np.ndarray,
        mask: Path | np.ndarray = None,
    ) -> dict:
        # Load image as grayscale
        if isinstance(image, Path):
            img = cv2.imread(str(image))
        # else:
        #     img = image.copy()
        # img = contrast_enhancement(img)  # per migliorare la detection dei contorni

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()
        #     img = crop_melanoma(img, mask_img)

        # Apply Canny edge detection algorithm
        edges = cv2.Canny(img, 100, 200)

        # Compute edge features
        num_edges = np.sum(edges == 255)
        edge_density = num_edges / (img.shape[0] * img.shape[1])

        # Find contours in binary mask
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

        # Compute mean and standard deviation of contour lengths
        contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
        if len(contour_lengths) > 0:
            mean_contour_length = np.mean(contour_lengths)
            std_contour_length = np.std(contour_lengths)
        else:
            mean_contour_length = 0
            std_contour_length = 0

        # Create dictionary of edge features
        num_pixels = img.shape[0] * img.shape[1]
        edge_features = {
            "number_of_edges": num_edges / num_pixels,
            "edge_density": edge_density / num_pixels,
            "mean_length_of_edges": mean_contour_length / num_pixels,
            "std_length_of_edges": std_contour_length / num_pixels,
        }

        return edge_features


    def extract_hog_features(
        image: Path | np.ndarray,
        mask: Path | np.ndarray = None,
    ) -> dict:
        if isinstance(image, Path):
            img = cv2.imread(str(image))
        # else:
        #     img = image.copy()
        # img = contrast_enhancement(img)

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()
        #     img = crop_melanoma(img, mask_img)

        # Define HOG parameters
        cell_size = (8, 8)
        block_size = (2, 2)
        num_bins = 9

        # Compute HOG features
        hog_features, hog_image = hog(
            img,
            orientations=num_bins,
            pixels_per_cell=cell_size,
            cells_per_block=block_size,
            block_norm="L2-Hys",
            visualize=True,
        )

        # Normalize HOG features by the total number of cells in the image
        height, width = img.shape[:2]
        num_cells_height = height // cell_size[0]
        num_cells_width = width // cell_size[1]
        total_cells = num_cells_height * num_cells_width
        hog_features /= total_cells

        # Create dictionary to store HOG features
        hog_features = {"HOG_features": hog_features}

        return hog_features


    def extract_lbp_features(
        image: Path | np.ndarray,
        mask: Path | np.ndarray = None,
        n_points: int = 8,
        radius: int = 3,
    ) -> dict:
        if isinstance(image, Path):
             img = cv2.imread(str(image))
        # else:
        #     img = image.copy()
        # img = contrast_enhancement(img)

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()
        #     img = crop_melanoma(img, mask_img)

        # Compute LBP features
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")

        # Compute histogram of LBP codes
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float") / (img.shape[0] * img.shape[1])

        # Create dictionary to store LBP features
        lbp_dict = {}
        for i in range(n_bins):
            lbp_dict["LBP_{:03d}".format(i)] = hist[i]

        return lbp_dict


    def extract_fourier_features(
        image: Path | np.ndarray, mask: Path | np.ndarray = None
    ) -> dict:
        if isinstance(image, Path):
            img = cv2.imread(str(image))
        # else:
        #     img = image.copy()
        # img = contrast_enhancement(img)

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()
        #     img = crop_melanoma(img, mask_img)

        # Compute Fourier Transform
        fft_image = fftshift(fft2(img))

        # Compute power spectrum
        power_spectrum = np.abs(fft_image) ** 2

        # Compute normalized power spectrum
        norm_power_spectrum = power_spectrum / np.sum(power_spectrum)

        # Compute moments of normalized power spectrum
        moments = []
        for i in range(4):
            for j in range(4):
                moment = np.sum(
                    (np.arange(norm_power_spectrum.shape[0])[:, np.newaxis] ** i)
                    * (np.arange(norm_power_spectrum.shape[1])[np.newaxis, :] ** j)
                    * norm_power_spectrum
                )
                moments.append(moment)

        # Create dictionary to store Fourier Transform features
        fourier_dict = {}
        for i, moment in enumerate(moments):
            fourier_dict["fourier_{:02d}".format(i)] = moment

        return fourier_dict


    def extract_haralick_features(
        image: Path | np.ndarray, mask: Path | np.ndarray = None
    ) -> dict:
        if isinstance(image, Path):
            img = cv2.imread(str(image))
        # else:
        #     img = image.copy()
        # img = contrast_enhancement(img)

        # if mask is not None:
        #     if isinstance(mask, Path):
        #         mask_img = cv2.imread(str(mask))
        #         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         mask_img = mask.copy()
        #     img = crop_melanoma(img, mask_img)

        # Compute Haralick features
        haralick_features = haralick(image)

        # Compute mean, standard deviation, skewness, and kurtosis of each Haralick feature
        haralick_mean = np.mean(haralick_features, axis=0)
        haralick_std = np.std(haralick_features, axis=0)
        haralick_skewness = np.zeros_like(haralick_mean)
        haralick_kurtosis = np.zeros_like(haralick_mean)
        n_features = haralick_features.shape[1]
        for i in range(n_features):
            haralick_skewness[i] = (
                np.mean((haralick_features[:, i] - haralick_mean[i]) ** 3)
                / haralick_std[i] ** 3
            )
            haralick_kurtosis[i] = (
                np.mean((haralick_features[:, i] - haralick_mean[i]) ** 4)
                / haralick_std[i] ** 4
                - 3
            )

        # Create dictionary to store Haralick features
        haralick_dict = {}
        feature_labels = [
            "angular_second_moment",
            "contrast",
            "correlation",
            "sum_of_squares_variance",
            "inverse_difference_moment",
            "sum_average",
            "sum_variance",
            "sum_entropy",
            "entropy",
            "difference_variance",
            "difference_entropy",
            "information_measure_of_correlation_1",
            "information_measure_of_correlation_2",
        ]

        for i in range(n_features):
            haralick_dict[feature_labels[i] + "_mean"] = haralick_mean[i]
            haralick_dict[feature_labels[i] + "_std"] = haralick_std[i]
            haralick_dict[feature_labels[i] + "_skewness"] = haralick_skewness[i]
            haralick_dict[feature_labels[i] + "_kurtosis"] = haralick_kurtosis[i]

        return haralick_dict


    


    if __name__ == "__main__":
        pass
