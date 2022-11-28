import os
import cv2
import math
import numpy as np
from zigzag import *
from Threads import *
from PIL import Image
import cv2
from runLengthEncoding import *

# defining block size
block_size = 8

global encodedSize, CodedbitsSize

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

    nbh = math.ceil(h / block_size)
    nbh = np.int32(nbh)

    nbw = math.ceil(w / block_size)
    nbw = np.int32(nbw)

    # Pad the image, because sometime image size is not dividable to block size
    # get the size of padded image by multiplying block size by number of blocks in height/width

    # height of padded image
    H = block_size * nbh

    # width of padded image
    W = block_size * nbw

    # create a numpy zero matrix with size of H,W
    padded_img = np.zeros((H, W))
    padded_img2 = np.zeros((H, W))
    padded_img3 = np.zeros((H, W))

    # or this other way here
    padded_img[0:height, 0:width] = b[0:height, 0:width]
    padded_img2[0:height, 0:width] = g[0:height, 0:width]
    padded_img3[0:height, 0:width] = r[0:height, 0:width]

    # start encoding:
    # divide image into block size by block size (here: 8-by-8) blocks
    # To each block apply 2D discrete cosine transform
    # reorder DCT coefficients in zig-zag order
    # reshaped it back to block size by block size (here: 8-by-8)

    for i in range(nbh):

        # Compute start and end row index of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            # Compute start & end column index of the block
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            rBlock = padded_img3[row_ind_1: row_ind_2, col_ind_1: col_ind_2]
            gBlock = padded_img2[row_ind_1: row_ind_2, col_ind_1: col_ind_2]
            bBlock = padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2]

            # apply 2D discrete cosine transform to the selected block
            rDCT = cv2.dct(rBlock)
            gDCT = cv2.dct(gBlock)
            bDCT = cv2.dct(bBlock)

            DCT_normalized_r = np.divide(rDCT, QUANTIZATION_MAT).astype(int)
            DCT_normalized_g = np.divide(gDCT, QUANTIZATION_MAT).astype(int)
            DCT_normalized_b = np.divide(bDCT, QUANTIZATION_MAT).astype(int)

            # reorder DCT coefficients in zig zag order by calling zigzag function
            # it will give you a one dimentional array
            reorderedR = zigzag(DCT_normalized_r)
            reorderedG = zigzag(DCT_normalized_g)
            reorderedB = zigzag(DCT_normalized_b)

            # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
            reshapedR = np.reshape(reorderedR, (block_size, block_size))
            reshapedG = np.reshape(reorderedG, (block_size, block_size))
            reshapedB = np.reshape(reorderedB, (block_size, block_size))

            # copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2] = reshapedB
            padded_img2[row_ind_1: row_ind_2, col_ind_1: col_ind_2] = reshapedG
            padded_img3[row_ind_1: row_ind_2, col_ind_1: col_ind_2] = reshapedR

    padded_image_merge = cv2.merge([padded_img, padded_img2, padded_img3])

    cv2.imwrite(saveAt + '/encoded.bmp', np.uint8(padded_image_merge))
    report.set(str(report.get()) + ("\n[+] Encoded image Saved .. "))

    # Calculating size of encoded image
    file_name2 = str(saveAt + '/encoded.bmp')
    file_stats = os.stat(file_name2)
    global encodedSize, CodedbitsSize
    encodedSize = (file_stats.st_size / (1024 * 1024))

    flattenImages = [str(padded_img.flatten().tolist()), str(padded_img2.flatten().tolist()),
                     str(padded_img3.flatten().tolist())]

    report.set(str(report.get()) + ("\n[+] Run length Encoding started .. "))
    t1 = ThreadWithReturnValue(target=run_length_encoding, args=(flattenImages[0],))
    t2 = ThreadWithReturnValue(target=run_length_encoding, args=(flattenImages[1],))
    t3 = ThreadWithReturnValue(target=run_length_encoding, args=(flattenImages[2],))

    t1.start()
    t2.start()
    t3.start()

    bitstream1 = t1.join()
    bitstream2 = t2.join()
    bitstream3 = t3.join()

    bitsream = bitstream1 + bitstream2 + bitstream3
    CodedbitsSize = ((len(bitsream) / 2) * 0.000000125)

    report.set(str(report.get()) + ("\n[+] Encoding complete .. "))

    return [[bitstream1, bitstream2, bitstream3], padded_img.shape]


def decodeImage(encodedText, h, w):
    # Decoding text
    imageR = run_length_decoding(encodedText)

    # Cleaning Results
    imageR = imageR[1:-1]
    res = imageR.split(',')
    res = np.array(res)
    res = res.astype(np.float)
    array = np.array([[res[i + j * w] for i in range(w)] for j in range(h)])

    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    # initialisation of compressed image
    padded_img = np.zeros((h, w))

    while i < h:
        j = 0
        while j < w:
            temp_stream = array[i:i + 8, j:j + 8]
            block = inverse_zigzag(temp_stream.flatten(), int(block_size), int(block_size))
            de_quantized = np.multiply(block, QUANTIZATION_MAT)
            padded_img[i:i + 8, j:j + 8] = cv2.idct(de_quantized)
            j = j + 8
        i = i + 8

    # clamping to  8-bit max-min values
    padded_img[padded_img > 255] = 255
    padded_img[padded_img < 0] = 0
    # compressed image is written into compressed_image.mp file
    return padded_img


def decodingDriver(encode, saveAt, report):
    H, W = encode[1]
    report.set(str(report.get()) + ("\n[+] Run length decoding started  .. "))
    t1 = ThreadWithReturnValue(target=decodeImage, args=(encode[0][0], H, W,))
    t2 = ThreadWithReturnValue(target=decodeImage, args=(encode[0][1], H, W))
    t3 = ThreadWithReturnValue(target=decodeImage, args=(encode[0][2], H, W,))

    report.set(str(report.get()) + ("\n[+] Threads Started .. "))

    t1.start()
    t2.start()
    t3.start()

    imageR = t1.join()
    imageG = t2.join()
    imageB = t3.join()

    # All threads completed
    report.set(str(report.get()) + ("\n[+] Decoded images buildup Complete .. "))
    report.set(str(report.get()) + ("\n[+] Merging decoded images .. "))
    image = cv2.merge([imageR, imageG, imageB])
    cv2.imwrite(saveAt + '/final.bmp', np.uint8(image))

    # Final Reporting
    report.set(str(report.get()) + ("\n[+] Final Decoded image Saved .. "))
    report.set(str(report.get()) + ("\n\n[+]  " + 50 * "-"))
    report.set(str(report.get()) + ("\n[+] Compression file size is " + str(np.round(CodedbitsSize, 1)) + " MB"))
    report.set(str(report.get()) + ("\n[+] Compression ratio is " + str(np.round(encodedSize / CodedbitsSize, 1))))
    report.set(str(report.get()) + ("\n[+] Compression percentage is " + str(np.round((CodedbitsSize/encodedSize)*100, 1))))
