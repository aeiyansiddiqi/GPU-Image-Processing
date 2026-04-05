import sys
import numpy as np
import warp as wp
from PIL import Image
# Just going to try to follow the steps from lecture 14

#1. Initialize Warp
wp.init()
device = "cpu"

#2. Load the Image
def loadImage(inputFile):
    img = Image.open(inputFile)
    #3. Get the image mode
    if img.mode != "L":
        img = img.convert("RGB")
        channels = 3
    else:
        channels = 1

    #Change to NumPy array
    imgNp = np.array(img, dtype=float)
    height, width = imgNp.shape[:2]

    # Warp arrays for channels
    originalImg = []
    output = []
    if channels == 3:
        # 5. Convert a NumPy array into Warp arrays
        for c in range(3):
            flattenedChannel = imgNp[..., c].flatten()
            originalImg.append(wp.array(flattenedChannel, dtype=float, device=device))
            # 6. Create an empty Warp array for the output image and initialize it to zero
            output.append(wp.zeros_like(originalImg[c], device=device))
    else:
        flattenedImage = imgNp.flatten()
        originalImg.append(wp.array(flattenedImage, dtype=float, device=device))
        output.append(wp.zeros_like(originalImg[0], device=device))

    return originalImg, output, height, width, channels

    


#w(i,j)=e^((-i^2+j^2​)/2σ^2)
def createGaussianKernel(kernelSize, sigma):
    @wp.kernel
    def gaussianKernel(kernelArray: wp.array(dtype=float)):  
        sum = 0.0
        for i in range(kernelSize):
            for j in range(kernelSize):
                x = float(i - (kernelSize // 2))
                y = float(j - (kernelSize // 2))

                #w(i,j)=e^((-i^2+j^2​)/2σ^2)
                value = wp.exp(-(x*x + y*y) / (2.0 * sigma * sigma))

                kernelArray[(i * kernelSize) + j] = value
                sum += value
        
        for kernelIndex in range(kernelSize * kernelSize):
            kernelArray[kernelIndex] = kernelArray[kernelIndex] / sum
    
    return gaussianKernel


@wp.kernel
def gaussianBlurKernel(inputImg: wp.array(dtype=float), 
                       outputImg: wp.array(dtype=float), 
                       kernel: wp.array(dtype=float),
                       kernelSize: int,
                       width: int,
                       height: int):
    tid = wp.tid()
    x = tid % width
    y = tid // width
    outputVal = float(0.0)

    for j in range(-(kernelSize//2), (kernelSize//2)+1):
        for i in range(-(kernelSize//2), (kernelSize//2)+1):
            imgX = x + i
            imgY = y + j
            # Reflective 
            if imgX < 0:
                imgX = -imgX - 1
            elif imgX >= width:
                imgX = 2*width - imgX - 1
            if imgY < 0:
                imgY = -imgY - 1
            elif imgY >= height:
                imgY = 2*height - imgY - 1
            imgIndex = imgY * width + imgX
            kernelIndex = (j + (kernelSize//2)) * kernelSize + (i + (kernelSize//2))
            outputVal += inputImg[imgIndex] * kernel[kernelIndex]
    outputImg[tid] = outputVal

def getGaussianImage(inputImg: wp.array(dtype=float), 
                     outputImg: wp.array(dtype=float), 
                     height: int, 
                     width: int, 
                     kernelSize: int, 
                     sigma: float):

    kernel = wp.zeros(kernelSize * kernelSize, dtype=float, device=device)

    gaussianKernelFunc = createGaussianKernel(kernelSize, sigma)
    wp.launch(gaussianKernelFunc, dim=1, inputs=[kernel], device=device)
    wp.synchronize_device()

    numPixels = height*width
    wp.launch(gaussianBlurKernel,
              dim=numPixels,
              inputs=[inputImg, outputImg, kernel, kernelSize, width, height],
              device=device)
    wp.synchronize_device()
    return outputImg


@wp.kernel
def unsharpMaskKernel(inputImg: wp.array(dtype=float),
                      blurred: wp.array(dtype=float),
                      output: wp.array(dtype=float),
                      k: float):
    tid = wp.tid()
    output[tid] = inputImg[tid]+k*(inputImg[tid] - blurred[tid])


def getSharpenedImage(inputImg: wp.array(dtype=float), 
                      blurred: wp.array(dtype=float),
                      k: float):
    output = wp.zeros_like(inputImg, device=device)
    numPixels = inputImg.shape[0]
    wp.launch(unsharpMaskKernel, dim=numPixels,
              inputs=[inputImg, blurred, output, k],
              device=device)
    wp.synchronize_device()
    return output


if __name__ == "__main__":

    #args
    userChoice = sys.argv[1]           
    kernelSize = int(sys.argv[2])   
    param = float(sys.argv[3])      
    inFileName = sys.argv[4]
    outFileName = sys.argv[5]

    
    originalImg, outputImg, height, width, channels = loadImage(inFileName)

    if userChoice == "-n":  
        resultImg = []
        for c in range(channels):
            blurred = getGaussianImage(originalImg[c], outputImg[c], height, width, kernelSize, param)
            resultImg.append(blurred)

    elif userChoice == "-s":  
        blurredImg = []
        for c in range(channels):
            blurred = getGaussianImage(originalImg[c], outputImg[c], height, width, kernelSize, param)
            blurredImg.append(blurred)

        # nsharp mask per channel
        resultImg = []
        for c in range(channels):
            sharpened = getSharpenedImage(originalImg[c], blurredImg[c], param)
            resultImg.append(sharpened)

    # warp to nump
    if channels == 3:
        resultNp = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            RGBInfo = np.asarray(resultImg[c].numpy(), dtype=np.uint8)
            resultNp[:, :, c] = RGBInfo.reshape(height, width)
    else:
        resultNp = np.asarray(resultImg[0].numpy(), dtype=np.uint8)
        resultNp = resultNp.reshape(height, width)

    # Save the processed image
    Image.fromarray(resultNp).save(outFileName)
