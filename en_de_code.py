import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from zigzag import *
from Threads import *
from PIL import Image
import cv2
import cv2 as cv
from runLengthEncoding import *
import PSNR

# defining block size
blockSize = 8

global encodedSize, CodedbitsSize
imgOrgPath =None

# Quantization Matrix
QUANTIZATION_MAT = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])


# If resizing needed then use this code
# def ImgResize(image, resize):
#     global imagePath
#     imagePath = image
#     try:
#         img1 = cv2.imread(image, cv2.IMREAD_COLOR)
#         b, g, r = cv2.split(img1)
#     except:
#         img = Image.open(image)
#         img1 = np.array(img)
#         b, g, r = cv2.split(img1)
#
#     [h, w] = b.shape
#
#     h = int(h * resize)
#     w = int(w * resize)
#
#     b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
#     g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
#     r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
#
#     return cv2.merge([b, g, r])


# Encoding is being done here
def encodingDriver(image, saveAt, report):
    try:
        img1 = cv2.imread(image, cv2.IMREAD_COLOR)

    except:
        img = Image.open(image)
        img1 = np.array(img)


    global imgOrgPath
    imgOrgPath = image

    report.set(str(report.get()) + ("\n[+] Encoding Started:" + str("")))

    b, g, r = cv2.split(img1)

    report.set(str(report.get()) + ("\n[+] Image Split .. "))

    # get size of the image
    [h, w] = b.shape

    # No of blocks needed : Calculation
    height = h
    width = w
    h = np.float32(h)
    w = np.float32(w)

    nbh = math.ceil(h / blockSize)
    nbh = np.int32(nbh)

    nbw = math.ceil(w / blockSize)
    nbw = np.int32(nbw)

    # Pad the image, because sometime image size is not dividable to block size
    # get the size of padded image by multiplying block size by number of blocks in height/width

    # height of padded image
    H = blockSize * nbh

    # width of padded image
    W = blockSize * nbw

    # create a numpy zero matrix with size of H,W
    paddedImg = np.zeros((H, W))
    paddedImg2 = np.zeros((H, W))
    paddedImg3 = np.zeros((H, W))

    # or this other way here
    paddedImg[0:height, 0:width] = b[0:height, 0:width]
    paddedImg2[0:height, 0:width] = g[0:height, 0:width]
    paddedImg3[0:height, 0:width] = r[0:height, 0:width]

    # start encoding:
    # divide image into block size by block size (here: 8-by-8) blocks
    # To each block apply 2D discrete cosine transform
    # reorder DCT coefficients in zig-zag order
    # reshaped it back to block size by block size (here: 8-by-8)

    for i in range(nbh):

        # Compute start and end row index of the block
        rowInd1 = i * blockSize
        rowInd2 = rowInd1 + blockSize

        for j in range(nbw):
            # Compute start & end column index of the block
            colInd1 = j * blockSize
            colInd2 = colInd1 + blockSize

            rBlock = paddedImg3[rowInd1: rowInd2, colInd1: colInd2]
            gBlock = paddedImg2[rowInd1: rowInd2, colInd1: colInd2]
            bBlock = paddedImg[rowInd1: rowInd2, colInd1: colInd2]

            # apply 2D discrete cosine transform to the selected block
            rDCT = cv2.dct(rBlock)
            gDCT = cv2.dct(gBlock)
            bDCT = cv2.dct(bBlock)

            rDCTNormalized = np.divide(rDCT, QUANTIZATION_MAT).astype(int)
            gDCTNormalized = np.divide(gDCT, QUANTIZATION_MAT).astype(int)
            bDCTNormalized = np.divide(bDCT, QUANTIZATION_MAT).astype(int)

            # reorder DCT coefficients in zig zag order by calling zigzag function
            # it will give you a one dimentional array
            reorderedR = zigzag(rDCTNormalized)
            reorderedG = zigzag(gDCTNormalized)
            reorderedB = zigzag(bDCTNormalized)

            # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
            reshapedR = np.reshape(reorderedR, (blockSize, blockSize))
            reshapedG = np.reshape(reorderedG, (blockSize, blockSize))
            reshapedB = np.reshape(reorderedB, (blockSize, blockSize))

            # copy reshaped matrix into paddedImg on current block corresponding indices
            paddedImg[rowInd1: rowInd2, colInd1: colInd2] = reshapedB
            paddedImg2[rowInd1: rowInd2, colInd1: colInd2] = reshapedG
            paddedImg3[rowInd1: rowInd2, colInd1: colInd2] = reshapedR

    paddedImageMerge = cv2.merge([paddedImg, paddedImg2, paddedImg3])

    cv2.imwrite(saveAt + '/encoded.bmp', np.uint8(paddedImageMerge))
    report.set(str(report.get()) + ("\n[+] Encoded image Saved .. "))

    # Calculating size of encoded image
    file_name2 = str(saveAt + '/encoded.bmp')
    file_stats = os.stat(file_name2)
    global encodedSize, CodedbitsSize
    encodedSize = (file_stats.st_size / (1024 * 1024))

    flattenImages = [str(paddedImg.flatten().tolist()), str(paddedImg2.flatten().tolist()),
                     str(paddedImg3.flatten().tolist())]

    report.set(str(report.get()) + ("\n[+] Run length Encoding started .. "))
    t1 = ThreadWithReturnValue(target=runLengthEncoding, args=(flattenImages[0],))
    t2 = ThreadWithReturnValue(target=runLengthEncoding, args=(flattenImages[1],))
    t3 = ThreadWithReturnValue(target=runLengthEncoding, args=(flattenImages[2],))

    # Started the threads
    t1.start()
    t2.start()
    t3.start()

    bitstream1 = t1.join()
    bitstream2 = t2.join()
    bitstream3 = t3.join()

    # Adding the three different channels into one variable
    bitsream = bitstream1 + bitstream2 + bitstream3
    # Converting the bitstream from bites to MB.
    # 24 equals the bytes of the three channels and chanel each equates 8 bytes
    # 1000000 used to convert from bytes to MB
    CodedbtsSize = ((len(bitsream) / 24) * 1000000)
    print(CodedbtsSize)


    CodedbitsSize = ((len(bitsream) / 2) * 0.000000125)
    report.set(str(report.get()) + ("\n[+] Encoding complete .. "))

    return [[bitstream1, bitstream2, bitstream3], paddedImg.shape]


def decodeImage(encodedText, h, w):
    # Decoding text
    imageR = runLengthDecoding(encodedText)

    # Cleaning Results
    imageR = imageR[1:-1]
    res = imageR.split(',')
    res = np.array(res)
    res = res.astype(np.float)
    array = np.array([[res[i + j * w] for i in range(w)] for j in range(h)])

    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    # initialisation of compressed image
    paddedImg = np.zeros((h, w))

    while i < h:
        j = 0
        while j < w:
            tempStream = array[i:i + 8, j:j + 8]
            block = inverseZigzag(tempStream.flatten(), int(blockSize), int(blockSize))
            deQuantized = np.multiply(block, QUANTIZATION_MAT)
            paddedImg[i:i + 8, j:j + 8] = cv2.idct(deQuantized)
            j = j + 8
        i = i + 8

    # clamping to  8-bit max-min values
    paddedImg[paddedImg > 255] = 255
    paddedImg[paddedImg < 0] = 0
    # compressed image is written into compressed_image.mp file
    return paddedImg


def decodingDriver(encode, saveAt, report):
    # Getting the height and the width of the image
    H, W = encode[1]

    # Started the run length encoding and each chanel runs on the compression
    report.set(str(report.get()) + ("\n[+] Run length decoding started  .. "))
    t1 = ThreadWithReturnValue(target=decodeImage, args=(encode[0][0], H, W,))
    t2 = ThreadWithReturnValue(target=decodeImage, args=(encode[0][1], H, W))
    t3 = ThreadWithReturnValue(target=decodeImage, args=(encode[0][2], H, W,))
    report.set(str(report.get()) + ("\n[+] Threads Started .. "))

    # Started the threads
    t1.start()
    t2.start()
    t3.start()


    imageR = t1.join()
    imageG = t2.join()
    imageB = t3.join()


    cv2.imwrite(saveAt + '/aR.bmp', np.uint8(imageR))
    cv2.imwrite(saveAt + '/aG.bmp', np.uint8(imageG))
    cv2.imwrite(saveAt + '/aB.bmp', np.uint8(imageB))

    # All threads completed
    report.set(str(report.get()) + ("\n[+] Decoded images buildup Complete .. "))
    report.set(str(report.get()) + ("\n[+] Merging decoded images .. "))
    image = cv2.merge([imageR, imageG, imageB])
    cv2.imwrite(saveAt + '/final.bmp', np.uint8(image))
    report.set(str(report.get()) + ("\n[+] Final Decoded image Saved .. "))

    # Final Reporting
    report.set(str(report.get()) + ("\n\n\n"))
    report.set(str(report.get()) + ("\n[+] METADATA "))
    report.set(str(report.get()) + ("\n[+] "))
    report.set(str(report.get()) + ("\n[+] Compression file size is " + str(np.round(CodedbitsSize, 1)) + " MB"))
    report.set(str(report.get()) + ("\n[+] Compression ratio is " + str(np.round(encodedSize / CodedbitsSize, 1))))
    report.set(str(report.get()) + ("\n[+] Compression percentage is " + str(np.round((CodedbitsSize/encodedSize)*100, 1))))
    report.set(str(report.get()) + ("\n[+] MSE is " + str(PSNR.MSE(imgOrgPath, saveAt + '/final.bmp'))))
    report.set(str(report.get()) + ("\n[+] PSNR is " + str(PSNR.PSNR(imgOrgPath,saveAt + '/final.bmp'))))