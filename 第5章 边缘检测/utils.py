from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns


class conv_2d():
    def __init__(self, image, kernel):
        self.img = image
        self.k = kernel

    def plot(self):
        # 展示输入图像
        plt.imshow(self.img[:, :, ::-1])
        plt.axis('off')
        plt.title('Input')
        plt.show()

        # 展示卷积核
        fig = plt.figure(figsize=(2, 1.5))
        sns.heatmap(self.k)
        plt.axis('off')
        plt.title('Kernel')

    # 定义二维卷积
    def convolution(self, data, k):
        # 直接调用库函数进行卷积操作
        return cv2.filter2D(data, -1, k)

    # 展示二维卷积结果
    def plot_conv(self):
        # 卷积过程
        img_new = self.convolution(self.img, self.k)
        # 卷积结果可视化            
        plt.figure()
        plt.imshow(img_new[:, :, ::-1])
        plt.title('Output')
        plt.axis('off')
        return


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    
    k = size // 2
    if sigma == 0:
        sigma = ((size - 1) * 0.5 - 1) * 0.3 + 0.8
    
    s = 2 * (sigma**2)
    sum_val = 0
    for i in range(0, size):
        for j in range(0, size):
            x = i - k
            y = j - k
            kernel[i,j] = np.exp(-(x**2+y**2) / s) / s / np.pi
    return kernel

def partial_x(img):
    Hi, Wi = img.shape  
    out = np.zeros((Hi, Wi))
    k = np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]])
    out = cv2.filter2D(img, -1, k)
    return out


def partial_y(img):
    Hi, Wi = img.shape  
    out = np.zeros((Hi, Wi))
    k = np.array([[0,0.5,0],[0,0,0],[0,-0.5,0]])
    out = cv2.filter2D(img, -1, k)
    return out


def non_maximum_suppression(G, theta):
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    theta %= 360
    out=G.copy()
    for i in range(1,H-1):
        for j in range(1,W-1):
            angle = theta[i,j]
            if angle==0 or angle==180:
                ma=max(G[i,j-1],G[i,j+1])
            elif angle==45 or angle==45+180:
                ma=max(G[i-1,j-1],G[i+1][j+1])
            elif angle==90 or angle==90+180:
                ma=max(G[i-1,j],G[i+1,j])
            elif angle==135 or angle==135+180:
                ma=max(G[i-1,j+1],G[i+1,j-1])
            else:
                print(angle)
                raise
            if ma>G[i,j]:
                out[i,j]=0
    return out


def double_thresholding(img, high, low):
    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)
    a,b = img.shape
    for i in range(a):
        for j in range(b):
            if img[i,j]>high:
                strong_edges[i,j] = 1
            elif img[i,j]<=high and img[i,j]>low:
                weak_edges[i,j] = 1

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    neighbors = []
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    
    q = [(i,j) for i in range(H) for j in range(W) if strong_edges[i,j]]
    while q:
        i,j=q.pop()
        for a,b in get_neighbors(i,j,H,W):
            if weak_edges[a][b]:
                weak_edges[a][b] = 0
                edges[a][b] = 1
                q.append((a,b))
                
    return edges


def canny(img, kernel_size=5, sigma=1.4, high=6, low=3):
    
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = cv2.filter2D(img, -1, kernel)    
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge


import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt
import random


def generateGimage(image, sigma, num_layers=8, k_stride=2):
    '''
    先生成指定数量的高斯核，在对image进行高斯模糊，得到scale space
    '''
    sigma_res = np.sqrt(np.max([(sigma ** 2) - ((1) ** 2), 0.01]))
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_res, sigmaY=sigma_res)

    # 生成高斯核
    k = 2 ** (1 / k_stride)
    gaussian_kernels = np.zeros(num_layers)
    # 第一层高斯就是1.6
    gaussian_kernels[0] = sigma

    for i in range(1, num_layers):
        # 根据高斯的性质，可以将大的高斯核拆分，减少计算量
        gaussian_old = k ** (i - 1) * sigma
        gaussian_new = k * gaussian_old
        gaussian_kernels[i] = np.sqrt(gaussian_new ** 2 - gaussian_old ** 2)

    # 进行高斯模糊
    gaussian_images = [image]
    for kernel in gaussian_kernels:
        tmp_image = cv2.GaussianBlur(image, (0, 0), sigmaX=kernel, sigmaY=kernel)
        gaussian_images.append(tmp_image)

    return np.array(gaussian_images)


def generateDoGSpace(gaussian_images):
    """
    生成dog空间
    """
    dog_images = []
    for img1, img2 in zip(gaussian_images, gaussian_images[1:]):
        dog_images.append(img2 - img1)
    return dog_images


def isLocalExtremum(l1, l2, l3, threshold):
    if l2[1, 1] > threshold:
        if l2[1, 1] > 0:
            return np.all(l2[1, 1] >= l1) and np.all(l2[1, 1] >= l3) and np.sum(l2[1, 1] < l2) == 0
        elif l2[1, 1] < 0:
            return np.all(l2[1, 1] <= l1) and np.all(l2[1, 1] <= l3) and np.sum(l2[1, 1] > l2) == 0
        return False


def compute1derivative(cube):
    """
    计算得到一阶导数
    """
    dx = (cube[1, 1, 2] - cube[1, 1, 0]) / 2
    dy = (cube[1, 2, 1] - cube[1, 0, 1]) / 2
    ds = (cube[2, 1, 1] - cube[0, 1, 1]) / 2
    return np.array([dx, dy, ds])


def compute2derivative(cube):
    """
    计算二阶导数
    """
    center = cube[1, 1, 1]
    dxx = cube[1, 1, 2] - 2 * center + cube[1, 1, 0]
    dxy = (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0]) / 4
    dxs = (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0]) / 4
    dyy = cube[1, 2, 1] - 2 * center + cube[1, 0, 1]
    dys = (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1]) / 4
    dss = cube[2, 1, 1] - 2 * center + cube[0, 1, 1]
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])


def computeOrien(pt, size, layer, g_image):
    # 1.5倍的3sigma原则，决定圆的大小
    radius = int(round(3 * size * 1.5))
    image_shape = g_image.shape
    num_bins = 36
    histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)
    orien_list = []

    for i in range(-radius, radius + 1):
        y = int(round(pt[1])) + i
        if y > 0 and y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                x = int(round(pt[0])) + j
                if x > 0 and x < image_shape[1] - 1:
                    dx = 0.5 * (g_image[y, x + 1] - g_image[y, x - 1])
                    dy = 0.5 * (g_image[y + 1, x] - g_image[y - 1, x])
                    value = np.sqrt(dx * dx + dy * dy)
                    orien = np.rad2deg(np.arctan2(dy, dx))
                    # 高斯加权
                    weight = np.exp(-0.5 / ((size * 1.5) ** 2) * (i ** 2 + j ** 2))
                    histogram_index = int(round(orien * num_bins / 360.))
                    histogram[histogram_index % num_bins] += weight * value

    for n in range(num_bins):
        # 高斯平滑
        smooth_histogram[n] = (6 * histogram[n] + 4 * (histogram[n - 1] + histogram[(n + 1) % num_bins]) + histogram[
            n - 2] + histogram[(n + 2) % num_bins]) / 16.

    orien_max = np.max(smooth_histogram)
    orien_local_max = list(i for i in range(len(smooth_histogram)) if
                           smooth_histogram[i] > smooth_histogram[i - 1] and smooth_histogram[i] > smooth_histogram[
                               (i + 1) % num_bins])
    # 大于最大值80%的，也要视为一个独立的blob
    for index in orien_local_max:
        if smooth_histogram[index] >= 0.8 * orien_max:
            orien_list.append(index * 360. / num_bins)

    return orien_list


def computeBlobAttribute(x, y, layer, dog_images, sigma, threshold, border, num_layers, g_image, corners, scales,
                         orientations, layers):
    gamma = 10
    image_shape = dog_images[0].shape
    out_flag = False

    '''
    使用牛顿迭代法得到极值点
    至多更新5次，如果未收敛，则认为该候选点不是极值点
    我们选择的极值点的坐标不需要量化，应当尽量避免量化误差
    '''
    for iter_num in range(5):
        img1, img2, img3 = dog_images[layer - 1:layer + 2]
        cube = np.array(
            [img1[x - 1:x + 2, y - 1:y + 2], img2[x - 1:x + 2, y - 1:y + 2], img3[x - 1:x + 2, y - 1:y + 2]])

        # 分别得到一二阶导数
        grad = compute1derivative(cube)
        hessian = compute2derivative(cube)
        # 解方程得到牛顿迭代的更新值
        update = -np.linalg.lstsq(hessian, grad, rcond=None)[0]
        # 如果移动的距离太小，说明当前点里极值已收敛，直接返回当前点即可
        if abs(update[0]) < 0.5 and abs(update[1]) < 0.5 and abs(update[2]) < 0.5:
            break
        # 更新当前点
        y += int(round(update[0]))
        x += int(round(update[1]))
        layer += int(round(update[2]))
        # 确保新的cube在space里
        if x < border or x >= image_shape[0] - border or y < border or y >= image_shape[
            1] - border or layer < 1 or layer > num_layers - 2:
            out_flag = True
            break

    # 超出scale space或者未不收敛，直接return
    if out_flag or iter_num >= 4:
        return

        # 使用公式计算极值点的对比度
    Extremum = cube[1, 1, 1] + 0.5 * np.dot(grad, update)
    # 确保当前点的对比度足够大
    if np.abs(Extremum) >= threshold:
        # 得到xy的hessian矩阵
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        # 特征值都为正，且最大最小特征值的比值不超过10，此时该点一般不为边缘
        if xy_hessian_det > 0 and (xy_hessian_trace ** 2) / xy_hessian_det < ((gamma + 1) ** 2) / gamma:
            pt = ((y + update[0]), (x + update[1]))
            size = sigma * (2 ** ((layer + update[2])))
            # 计算keypoint的方向，由于一个blob可能根据方向变成多个blob，所以返回值是list
            orien_list = computeOrien(pt, size, layer, g_image)
            for tmp_orien in orien_list:
                corners.append(pt)
                scales.append(size)
                layers.append(layer)
                orientations.append(tmp_orien)
    return


def tirlinearInterpolation(i, j, value, orien, result_cube):
    i_quant = int(np.floor(i))
    j_quant = int(np.floor(j))
    orien_quant = int(np.floor(orien)) % 8

    i_residual = i - i_quant
    j_residual = j - j_quant
    orien_residual = (orien - orien_quant) % 8

    c1 = i_residual * value
    c0 = (1 - i_residual) * value
    c11 = c1 * j_residual
    c10 = c1 * (1 - j_residual)
    c01 = c0 * j_residual
    c00 = c0 * (1 - j_residual)
    c111 = c11 * orien_residual
    c110 = c11 * (1 - orien_residual)
    c101 = c10 * orien_residual
    c100 = c10 * (1 - orien_residual)
    c011 = c01 * orien_residual
    c010 = c01 * (1 - orien_residual)
    c001 = c00 * orien_residual
    c000 = c00 * (1 - orien_residual)

    result_cube[i_quant + 1, j_quant + 1, orien_quant] += c000
    result_cube[i_quant + 1, j_quant + 1, (orien_quant + 1) % 8] += c001
    result_cube[i_quant + 1, j_quant + 2, orien_quant] += c010
    result_cube[i_quant + 1, j_quant + 2, (orien_quant + 1) % 8] += c011
    result_cube[i_quant + 2, j_quant + 1, orien_quant] += c100
    result_cube[i_quant + 2, j_quant + 1, (orien_quant + 1) % 8] += c101
    result_cube[i_quant + 2, j_quant + 2, orien_quant] += c110
    result_cube[i_quant + 2, j_quant + 2, (orien_quant + 1) % 8] += c111

    return


def detect_blobs(image):
    """Laplacian blob detector.

    Args:
    - image (2D float64 array): A grayscale image.

    Returns:
    - corners (list of 2-tuples): A list of 2-tuples representing the locations
        of detected blobs. Each tuple contains the (x, y) coordinates of a
        pixel, which can be indexed by image[y, x].
    - scales (list of floats): A list of floats representing the scales of
        detected blobs. Has the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.
    """
    sigma = 1.6
    num_layers = 4
    border = 5
    k_stride = 1
    g_images = generateGimage(image, sigma, num_layers, k_stride)
    dog_images = generateDoGSpace(g_images)

    # 开始寻找blob
    threshold = 0.02
    corners = []
    scales = []
    orientations = []
    layers = []

    for layer, (image1, image2, image3) in enumerate(zip(dog_images, dog_images[1:], dog_images[2:])):
        # 忽略太靠近边缘的点
        for x in range(border, image1.shape[0] - border):
            for y in range(border, image2.shape[1] - border):
                # 检测当前位置是否为局部极值
                if isLocalExtremum(image1[x - 1:x + 2, y - 1:y + 2], image2[x - 1:x + 2, y - 1:y + 2],
                                   image3[x - 1:x + 2, y - 1:y + 2], threshold):
                    # 如果是备选点，就计算其各项属性
                    computeBlobAttribute(x, y, layer + 1, dog_images, sigma, threshold, border, num_layers,
                                         g_images[layer], corners, scales, orientations, layers)

    return g_images, corners, scales, orientations, layers


def compute_descriptors(g_images, corners, scales, orientations, layers):
    """Compute descriptors for corners at specified scales.

    Args:
    - image (2d float64 array): A grayscale image.
    - corners (list of 2-tuples): A list of (x, y) coordinates.
    - scales (list of floats): A list of scales corresponding to the corners.
        Must have the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.

    Returns:
    - descriptors (list of 1d array): A list of desciptors for each corner.
        Each element is an 1d array of length 128.
    """
    if len(corners) != len(scales) or len(corners) != len(orientations):
        raise ValueError(
            '`corners`, `scales` and `orientations` must all have the same length.')

    descriptors_list = []

    for pt, size, orien, layer in zip(corners, scales, orientations, layers):
        # 取得对应的高斯模糊图像进行梯度计算
        g_image = g_images[layer]
        # 读取blob的各项信息
        x, y = np.round(np.array(pt)).astype(np.int32)
        orien = 360 - orien
        '''
        根据论文计算各个窗口的大小
        小窗口边长为1.5sigma
        '''
        win_s = 1.5 * size
        win_l = int(
            round(min(2 ** 0.5 * win_s * ((4 + 1) / 2), np.sqrt(g_image.shape[0] ** 2 + g_image.shape[1] ** 2))))
        # 用list依次存储方框内所有点的信息
        i_index = []
        j_index = []
        value_list = []
        orien_index = []
        # 三维数组存储16个窗口的8个方向，多出来的2是作为border，用来计算三线性插值的
        result_cube = np.zeros((4 + 2, 4 + 2, 8))

        for i in range(-win_l, win_l + 1):
            for j in range(-win_l, win_l + 1):

                i_rotate = j * np.sin(np.deg2rad(orien)) + i * np.cos(np.deg2rad(orien))
                j_rotate = j * np.cos(np.deg2rad(orien)) - i * np.sin(np.deg2rad(orien))

                tmp_i = (i_rotate / win_s) + 2 - 0.5
                tmp_j = (j_rotate / win_s) + 2 - 0.5

                if tmp_i > -1 and tmp_j > -1 and tmp_i < 4 and tmp_j < 4:
                    i_inimg = int(round(y + i))
                    j_inimg = int(round(x + j))
                    if i_inimg > 0 and j_inimg > 0 and i_inimg < g_image.shape[0] - 1 and j_inimg < g_image.shape[
                        1] - 1:
                        dx = g_image[i_inimg, j_inimg + 1] - g_image[i_inimg, j_inimg - 1]
                        dy = g_image[i_inimg - 1, j_inimg] - g_image[i_inimg + 1, j_inimg]
                        grad_value = np.sqrt(dx ** 2 + dy ** 2)
                        grad_orien = np.rad2deg(np.arctan2(dy, dx)) % 360
                        i_index.append(tmp_i)
                        j_index.append(tmp_j)
                        g_weight = np.exp(-1 / 8 * ((i_rotate / win_s) ** 2 + (j_rotate / win_s) ** 2))
                        value_list.append(g_weight * grad_value)
                        orien_index.append((grad_orien - orien) * 8 / 360)

        # 进行三线性插值
        for i, j, value, orien1 in zip(i_index, j_index, value_list, orien_index):
            tirlinearInterpolation(i, j, value, orien1, result_cube)

        # 归一化与截断
        descriptor = result_cube[1:-1, 1:-1, :].flatten()
        l2norm = np.linalg.norm(descriptor)
        threshold = l2norm * 0.2
        descriptor[descriptor > threshold] = threshold
        descriptor /= l2norm
        descriptors_list.append(descriptor)
    return descriptors_list


def match_descriptors(descriptors1, descriptors2):
    """Match descriptors based on their L2-distance and the "ratio test".

    Args:
    - descriptors1 (list of 1d arrays):
    - descriptors2 (list of 1d arrays):

    Returns:
    - matches (list of 2-tuples): A list of 2-tuples representing the matching
        indices. Each tuple contains two integer indices. For example, tuple
        (0, 42) indicates that corners1[0] is matched to corners2[42].
    """
    max_index = np.zeros((len(descriptors1))) - 1
    maxmatch = np.zeros((len(descriptors1))) + 1e10
    secmatch = np.zeros((len(descriptors1))) + 1e10
    threshold = 0.8

    for vec1_index in range(len(descriptors1)):
        for vec2_index in range(len(descriptors2)):
            distance = np.linalg.norm(descriptors1[vec1_index] - descriptors2[vec2_index])
            if distance < maxmatch[vec1_index]:
                maxmatch[vec1_index] = distance
                max_index[vec1_index] = vec2_index
            elif distance < secmatch[vec1_index]:
                secmatch[vec1_index] = distance

    matches = []
    for i in range(len(descriptors1)):
        if maxmatch[i] / secmatch[i] < threshold:
            matches.append((i, int(max_index[i])))

    return matches


def draw_matches(image1, image2, corners1, corners2, matches,
                 outlier_labels=None):
    """Draw matched corners between images.

    Args:
    - matches (list of 2-tuples)
    - image1 (3D uint8 array): A color image having shape (H1, W1, 3).
    - image2 (3D uint8 array): A color image having shape (H2, W2, 3).
    - corners1 (list of 2-tuples)
    - corners2 (list of 2-tuples)
    - outlier_labels (list of bool)

    Returns:
    - match_image (3D uint8 array): A color image having shape
        (max(H1, H2), W1 + W2, 3).
    """
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    hres = 0
    if h1 > h2:
        hres = int((h1 - h2) / 2)

    match_image = np.zeros((h1, w1 + w2, 3), np.uint8)

    for i in range(3):
        match_image[:h1, :w1, i] = image1
        match_image[hres:hres + h2, w1:w1 + w2, i] = image2

    for i in range(len(matches)):
        m = matches[i]
        pt1 = (int(corners1[m[0]][0]), int(corners1[m[0]][1]))
        pt2 = (int(corners2[m[1]][0] + w1), int(corners2[m[1]][1] + hres))
        cv2.circle(match_image, pt1, 1, (0, 255, 0), 2)
        cv2.circle(match_image, (pt2[0], pt2[1]), 1, (0, 255, 0), 2)
        if outlier_labels[i] == 1:
            cv2.line(match_image, pt1, pt2, (255, 0, 0))
        else:
            cv2.line(match_image, pt1, pt2, (0, 0, 255))

    return match_image


def compute_affine_xform(corners1, corners2, matches):
    """Compute affine transformation given matched feature locations.

    Args:
    - corners1 (list of 2-tuples)
    - corners1 (list of 2-tuples)
    - matches (list of 2-tuples)

    Returns:
    - xform (2D float64 array): A 3x3 matrix representing the affine
        transformation that maps coordinates in image1 to the corresponding
        coordinates in image2.
    - outlier_labels (list of bool): A list of Boolean values indicating whether
        the corresponding match in `matches` is an outlier or not. For example,
        if `matches[42]` is determined as an outlier match after RANSAC, then
        `outlier_labels[42]` should have value `True`.
    """
    iteration = 50
    M_list = []
    inlier_num_list = []
    for _ in range(iteration):
        sample_index = random.sample(range(len(matches)), 4)
        x1_s, y1_s = corners1[matches[sample_index[0]][0]]
        x1_t, y1_t = corners2[matches[sample_index[0]][1]]
        x2_s, y2_s = corners1[matches[sample_index[1]][0]]
        x2_t, y2_t = corners2[matches[sample_index[1]][1]]
        x3_s, y3_s = corners1[matches[sample_index[2]][0]]
        x3_t, y3_t = corners2[matches[sample_index[2]][1]]
        x4_s, y4_s = corners1[matches[sample_index[3]][0]]
        x4_t, y4_t = corners2[matches[sample_index[3]][1]]

        A = np.array([[x1_s, y1_s, 1, 0, 0, 0, -x1_t * x1_s, -x1_t * y1_s, -x1_t],
                      [0, 0, 0, x1_s, y1_s, 1, -y1_t * x1_s, -y1_t * y1_s, -y1_t],
                      [x2_s, y2_s, 1, 0, 0, 0, -x2_t * x2_s, -x2_t * y2_s, -x2_t],
                      [0, 0, 0, x2_s, y2_s, 1, -y2_t * x2_s, -y2_t * y2_s, -y2_t],
                      [x3_s, y3_s, 1, 0, 0, 0, -x3_t * x3_s, -x3_t * y3_s, -x3_t],
                      [0, 0, 0, x3_s, y3_s, 1, -y3_t * x3_s, -y3_t * y3_s, -y3_t],
                      [x4_s, y4_s, 1, 0, 0, 0, -x4_t * x4_s, -x4_t * y4_s, -x4_t],
                      [0, 0, 0, x4_s, y4_s, 1, -y4_t * x4_s, -y4_t * y4_s, -y4_t]
                      ])

        _, _, v = np.linalg.svd(A)
        M = np.reshape(v[-1], (3, 3))

        # M = t.reshape((3,3))
        inlier_num = 0
        for (index1, index2) in matches:
            coord1 = [corners1[index1][0], corners1[index1][1], 1]
            coord2 = [corners2[index2][0], corners2[index2][1], 1]
            mapcoor = np.dot(M, coord1)
            mapcoor = mapcoor / mapcoor[-1]
            if np.linalg.norm(coord2 - mapcoor) < 5:
                inlier_num += 1
        M_list.append(M)
        inlier_num_list.append(inlier_num)

    best_index = np.argmax(inlier_num_list)
    xform = M_list[best_index].astype(np.float64)
    outlier_labels = []
    for (index1, index2) in matches:
        coord1 = [corners1[index1][0], corners1[index1][1], 1]
        coord2 = [corners2[index2][0], corners2[index2][1], 1]
        mapcoor = np.dot(xform, coord1)
        mapcoor = mapcoor / mapcoor[-1]
        if np.linalg.norm(coord2 - mapcoor) < 12:
            outlier_labels.append(1)
        else:
            outlier_labels.append(0)

    return xform, outlier_labels


def stitch_images(image1, image2, xform):
    """Stitch two matched images given the transformation between them.

    Args:
    - image1 (3D uint8 array): A color image.
    - image2 (3D uint8 array): A color image.
    - xform (2D float64 array): A 3x3 matrix representing the transformation
        between image1 and image2. This transformation should map coordinates
        in image1 to the corresponding coordinates in image2.

    Returns:
    - image_stitched (3D uint8 array)
    """
    new_image = cv2.warpPerspective(image1, xform, (image2.shape[1], image2.shape[0]))
    image = new_image.copy()

    h, w, _ = image1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, xform)

    image = cv2.polylines(image2, [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
    image = np.where(new_image > 0, 0.5 * new_image + 0.5 * image, image)
    return image.astype(np.uint8)


def plot_image(img, name, camp='gray'):
    plt.imshow(img, cmap=camp)
    plt.title(name)
    plt.axis('off')
    plt.show()