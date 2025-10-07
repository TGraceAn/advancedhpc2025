import numba.cuda as cuda
from numba import config
import numba
config.CUDA_ENABLE_PYNVJITLINK = 0
import matplotlib.pyplot as plt
import numpy as np
import time

# ------ img ------ #
img = plt.imread('lena.jpeg')
plt.imshow(img)
h, w, c = img.shape

# save cause I'm on a remote server
plt.savefig('lena.png')

kernel_7x7 = np.array([[0, 0, 1, 2, 1, 0, 0],
                        [0, 3, 13, 22, 13, 3, 0],
                        [1, 13, 59, 97, 59, 13, 1],
                        [2, 22, 97, 159, 97, 22, 2],
                        [1, 13, 59, 97, 59, 13, 1],
                        [0, 3, 13, 22, 13, 3, 0],
                        [0, 0, 1, 2, 1, 0, 0]]) / 1003

@cuda.jit
def gaussian_blur_2D(src, dst, kernel):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    r = 0
    g = 0
    b = 0

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = tidx + i - center[0]
            y = tidy + j - center[1]
            if (0 <= x < src.shape[0]) and (0 <= y < src.shape[1]):
                r += src[x, y, 0] * kernel[i, j]
                g += src[x, y, 1] * kernel[i, j]
                b += src[x, y, 2] * kernel[i, j]
    dst[tidx, tidy, 0] = r
    dst[tidx, tidy, 1] = g
    dst[tidx, tidy, 2] = b


k_h, k_w = kernel_7x7.shape
@cuda.jit
def gaussian_blur_2D_mem_ker(src, dst, kernel):
    # shared for kernel memory
    tile = cuda.shared.array((k_h, k_w), kernel.dtype)
    # block to load kernel into shared mem
    tx = cuda.threadIdx.x # 0~(blockDim.x-1)
    ty = cuda.threadIdx.y # 0~(blockDim.y-1)
    # shared mem load
    tile[tx, ty] = kernel[tx, ty]
    # await all threads to load data
    cuda.syncthreads()
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    center = (tile.shape[0] // 2, tile.shape[1] // 2)
    r = 0
    g = 0
    b = 0
    for i in range(tile.shape[0]):
        for j in range(tile.shape[1]):
            x = tidx + i - center[0]
            y = tidy + j - center[1]
            if (0 <= x < src.shape[0]) and (0 <= y < src.shape[1]):
                r += src[x, y, 0] * tile[i, j]
                g += src[x, y, 1] * tile[i, j]
                b += src[x, y, 2] * tile[i, j]
    dst[tidx, tidy, 0] = r
    dst[tidx, tidy, 1] = g
    dst[tidx, tidy, 2] = b


@cuda.jit
def gaussian_blur_2D_mem_src(src, dst, kernel):
    # shared for kernel memory
    tile = cuda.shared.array((8+6, 8+6, 3), np.uint8) # for padding 6

    # block to load src into shared mem
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    # shared mem load
    tile[cuda.threadIdx.x, cuda.threadIdx.y, 0] = src[tidx, tidy, 0]
    tile[cuda.threadIdx.x, cuda.threadIdx.y, 1] = src[tidx, tidy, 1]
    tile[cuda.threadIdx.x, cuda.threadIdx.y, 2] = src[tidx, tidy, 2]

    # await all threads to load data
    cuda.syncthreads()
    center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    r = 0
    g = 0
    b = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = cuda.threadIdx.x + i - center[0]
            y = cuda.threadIdx.y + j - center[1]
            if (0 <= x < tile.shape[0]) and (0 <= y < tile.shape[1]):
                r += tile[x, y, 0] * kernel[i, j]
                g += tile[x, y, 1] * kernel[i, j]
                b += tile[x, y, 2] * kernel[i, j]
    dst[x, y, 0] = r
    dst[x, y, 1] = g
    dst[x, y, 2] = b

    

blockSize = (8, 8) # this is 64
gridSize = (int(h / blockSize[0]) + 1, int(w / blockSize[1]) + 1) # prevent unprocess pixels?

# --- main? --- #
# all the "variables" here are
# HOST feeds DEVICE with data
d_img = cuda.to_device(img)
d_kernel = cuda.to_device(kernel_7x7)

# destination on device
d_img_blur = cuda.device_array(img.shape, dtype=np.uint8)
# destination on host
img_blur = np.zeros((h, w, c), dtype=np.uint8)

# DEVICE process data in parallel
start = time.time()
gaussian_blur_2D[gridSize, blockSize](d_img, d_img_blur, d_kernel)
# DEVICE returns result to HOST
d_img_blur.copy_to_host(img_blur)
end = time.time()
# save image
plt.imshow(img_blur) 
plt.savefig('lena_gaussian_blur.png')
print(f'GPU Time 2D Gaussian Blur: {end-start}')


start = time.time()
gaussian_blur_2D_mem_ker[gridSize, blockSize](d_img, d_img_blur, d_kernel)
# DEVICE returns result to HOST
d_img_blur.copy_to_host(img_blur)
end = time.time()

plt.imshow(img_blur) 
plt.savefig('lena_gaussian_blur_mem_ker.png')
print(f'GPU Time 2D Kernel "Cache" Gaussian Blur: {end-start}')

start = time.time()
gaussian_blur_2D_mem_src[gridSize, blockSize](d_img, d_img_blur, d_kernel)
# DEVICE returns result to HOST
d_img_blur.copy_to_host(img_blur)
end = time.time()

plt.imshow(img_blur) 
plt.savefig('lena_gaussian_blur_mem_src.png')
print(f'GPU Time 2D Image "Cache" Gaussian Blur: {end-start}')



