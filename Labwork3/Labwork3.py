import numba.cuda as cuda
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 0
import matplotlib.pyplot as plt
import numpy as np
import time

# load img
img = plt.imread('hehehe.jpg')
plt.imshow(img)

# save cause I'm on a remote server
plt.savefig('hehehe.png')

print(img.shape)
h, w, c = img.shape

# reshape to 1D array
pixelCount = h*w
oneD_img = img.reshape(h*w, c)

# convert to int8 so that numba can handle
oneD_img = oneD_img.astype(np.uint8)

print(oneD_img.shape)


oneD_img2 = np.zeros((pixelCount, c), dtype=np.uint8)

r, g, b = oneD_img[:, 0], oneD_img[:, 1], oneD_img[:, 2]

# --- CPU version ---
start = time.time()
for i in range(pixelCount):
    gray = r[i]//3 + g[i]//3 + b[i]//3 # //3 to prevent overflow
    oneD_img2[i] = [gray, gray, gray]
end = time.time()
print(f'CPU Time: {end-start}')

oneD_img2 = oneD_img2.reshape(h, w, c)
plt.imshow(oneD_img2)
plt.savefig('hehehe_gray_cpu.png')

# --- CPU version, I dont like // so let's do smth else ---
oneD_img2 = np.zeros((pixelCount, c), dtype=np.uint8)

r, g, b = oneD_img[:, 0], oneD_img[:, 1], oneD_img[:, 2]

start = time.time()
for i in range(pixelCount):
    gray = int(r[i]/3) + int(g[i]/3) + int(b[i]/3)
    if gray > 255:
        gray = np.uint8(255)
    oneD_img2[i] = [gray, gray, gray]
end = time.time()
print(f'CPU Time: {end-start}')

oneD_img2 = oneD_img2.reshape(h, w, c)
plt.imshow(oneD_img2)
plt.savefig('hehehe_gray_cpu.png')

# --- GPU version ---
# # Host feeds device with data
# d_oneD_img = cuda.device_array(oneD_img.shape, dtype=np.uint8)

# to device
d_oneD_img = cuda.to_device(oneD_img)

print(d_oneD_img.shape)

# Host ask device to process data
@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

# define block size and grid size for kernel launch
blockSize = 64
gridSize = int(pixelCount / blockSize) + 1 # prevent unprocess pixels?
oneD_img2_gpu = np.zeros((pixelCount, c), dtype=np.uint8)

# gray image on device
d_oneD_img2 = cuda.device_array(oneD_img.shape, dtype=np.uint8)

# Device process data in parallel
start = time.time()
grayscale[gridSize, blockSize](d_oneD_img, d_oneD_img2)
# Device returns result to host 
d_oneD_img2.copy_to_host(oneD_img2_gpu)
end = time.time()

print(f'GPU Time: {end-start}')

# save result
oneD_img2_gpu = oneD_img2_gpu.reshape(h, w, c)
plt.imshow(oneD_img2_gpu)
plt.savefig('hehehe_gray_gpu.png')


# try different block sizes
two_hat = [2**i for i in range(1, 8)]

time_results = []

for blockSize in two_hat:
    value = 0
    # exp for 1000 times and take avg
    for _ in range(1000):
        gridSize = int(pixelCount / blockSize) + 1 # prevent unprocess pixels?
        oneD_img2_gpu = np.zeros((pixelCount, c), dtype=np.uint8)
        d_oneD_img2 = cuda.device_array(oneD_img.shape, dtype=np.uint8)
        start = time.time()
        grayscale[gridSize, blockSize](d_oneD_img, d_oneD_img2)
        d_oneD_img2.copy_to_host(oneD_img2_gpu)
        end = time.time()
        value += (end-start)
    time_results.append(value / 1000)

plt.figure()
plt.plot(two_hat, time_results)
plt.xticks(two_hat, labels=[str(x) for x in two_hat])
plt.xscale('log', base=2)
plt.xlabel('Block Size')
plt.ylabel('Time (s)')
plt.title('Block Size vs Time')
plt.savefig('block_size_vs_time.png')
