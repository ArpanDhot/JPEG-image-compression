# <a name="_bookmark0"></a>Abstract
There is no doubt that the modern world relies on large amounts of data in the form of images, videos, and other media types to carry out business, communicate, and connect with one another. In addition to the enormous amount of data that is generated, there is also the need for a large amount of storage to store all of this data, which can prove expensive as more data is generated. In most cases, it is primarily redundant data that humans cannot even perceive that can be as large as tens to hundreds of megabytes or gigabytes. This then raises the question of how this redundant data can be removed so that the size of the images can be smaller, thus saving on storage space and reducing the cost of the images. This question is answered by a process called image compression.



# <a name="_bookmark1"></a>Introduction
There are many types of data compression, but image compression is one that reduces the size of a digital image by compressing the data. The process of compressing an image may be lossy, which means that some data may be lost in the process of reducing the image size, and this lost information cannot be recovered during the decompression process. Compression can also be lossless, which maintains the data of an image throughout the whole process of compression and decompression. In the world of digital images, JPEG compression is one of the most popular lossy compression algorithms. The JPEG format is based on the Discrete Cosine Transform (DCT), which extracts spatial frequency information from spatial amplitude samples. Thereafter, the image would be quantised so that any information visually insignificant to the human eye would be removed from the image. Following this, encoding methods, such as Run-Length encoding, will be used in order to remove some more redundant information, thus achieving compression. When decompressing an image, the compression methods would have to be reversed in order for the image to revert to its original format.

This report will provide a detailed JPEG compression method and the steps taken to reach compression. We will show the methods we used, detailing the libraries, functions and data structures used to achieve the results. The results will then be analysed and discussed using different metrics. We will then end this report with a conclusion and any references and acknowledgements.


# <a name="_bookmark2"></a>Methodology
Below is an image of the steps taken in JPEG to reach compression as well as decompression. We will go into detail about each step, showing how we implemented the different stages and features.

![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.001.png)

# <a name="_bookmark3"></a>Compression
As part of this project, we use Python with libraries such as OpenCV, Math, and NumPy. Using OpenCV, we read an image without changing its content, such as its colour, using the imread() function. The image is then split into three channels and stored in three variables. Having chosen one channel, we first gauged the height and width of the image, so we could determine if our jpeg compression algorithm would allow us to divide it into even blocks based on the height and width of the image. The JPEG compression algorithm works by dividing images into blocks of nxn in order to compress them. A block size of 8x8 was chosen for this project because it is standard for JPEG. Alternatively, a larger block size, such as 32x32 or 64x64, could have been implemented, but the downside of using a larger block size is that the image will be less smooth over each block, which will reduce the compression rate of the image. It could have been possible to choose a smaller block size; however, this would have made the quantization step less flexible, resulting in little to no compression.

After dividing the image into 8x8 blocks, the next step would be to pad the image. When there are no multiples of 2 in the image dimensions, padding is necessary because sometimes this will result in part of the image being partially occupied by the image dimensions. These blocks at the boundaries must then be “padded” to the full 8x8 block size and then processed similarly to every other block. As part of our method, we used zero padding to make the image a Multiple of 8, which involves appending zeros to the end of the input sequence. Np.zeros() allowed us to achieve this desired effect.




|<p></p><p>![A picture containing text  Description automatically generated](Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.002.jpeg)</p>|
| :- |
|<p>This image shows the block preparation and image padding that needs to be done on the image, here we divided the image into 8x8 blocks. The white space represents the space that will get</p><p>padded by the zeroes.</p>|


The next step of JPEG compression is to apply Discrete Cosine Transform (DCT) to each block separately, where the output of each DCT is an 8x8 matrix of coefficients. DCT works by converting the spatial information into numeric (frequency information) data which can then be manipulated. The data is manipulated in the next step, called quantization; in this stage, less significant DCT coefficients are removed, which causes this method to be lossless. The DCT coefficients are divided by the data in the quantization matrix to obtain the quantized coefficients. Using zigzag scanning, we order the quantized coefficients with the higher energy at the top, followed by the lower energy and zero values at the bottom.

![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.003.png)Afterwards, Run-Length Encoding is applied to the data. Run-length encoding (RLE) method compresses images in a lossless manner, storing the redundant data in a sequence as a single value. The image can then be reconstructed during decompression exactly from this data.


|![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.004.png)|
| :-: |
|After encoding, we store the three channels' encoded data in the memory in the bitstream variable. As shown in the terminal, we next printed out the bitstream, which can be seen at the bottom of the image.|


In order to return to the original image, we have to decompress the encoded data after compressing the image. During the decompression process, the header information, as well as the quantisation factors, are removed. It is necessary to extract the data from the Run-Length encoded bit stream. In the next step, each coefficient is scaled using inverse quantisation, and the coefficients are then prepared for the inverse DCT by inverse quantisation. After the 8x8 pixel blocks are put into the image buffer, they are converted back to RGB images.

# <a name="_bookmark4"></a>Results and evaluations
Upon achieving the compression and decompression of the image, we can compare the size of the original image with the size of the compressed image. This difference can be seen in the table below. As a result, we can evaluate the data using a variety of metrics, including the mean square error and the peak signal-to- noise ratio. In decibels, a PSNR block computes the peak signal-to-noise ratio between two images. This ratio can compare the quality of original and compressed images


|No.|Original Size|Compressed Size|Compressed Ratio|<p>Compressed</p><p>Percentage</p>|MSE|PSNR|
| :- | :- | :- | :- | :- | :- | :- |
|[1](#_bookmark8)|34\.8 MB|18\.6 MB|1\.9|53\.3%|29\.44|33\.44|
|[2](#_bookmark8)|34\.8 MB|18\.5 MB|1\.9|53\.1%|17\.86|35\.61|
|[3](#_bookmark8)|34\.8 MB|18\.5 MB|1\.9|53\.0%|19\.11|35\.31|
|[4](#_bookmark8)|1\.17 MB|0\.5 MB|1\.9|52\.7%|4\.94|41\.18|

Images in the appendix and the data. Please click on the image number to go to the appendix.

The table above shows that the original images of all four images could be compressed by at least 50%, resulting in a consistent compression ratio of 1.9 for all four images. After evaluating the results, we used the mean squared error (MSE) and PSNR. The mean squared error measures the squared error between the original and compressed images. If the MSE is low, then there is a low error rate between the original and compressed images, which means that the image quality is relatively equal. In general, a higher PSNR indicates a better compression rate.



|![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.005.png)|![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.006.png)|
| :- | :- |
|<p>The dinosaur image was inputted as raw, so it has not been converted to greyscale, even though the visual is grey. This image got split into RGB channels, and JPEG compression was applied to every channel. Each channel is encoded using Run-Length encoding (RLE). Furthermore, as shown in the image above, the image on the right is reconstructed, which is of lower quality than the original image on the left. This is because RLE in grey-level images has constant intensity on</p><p>similar consecutive pixels. This results in a reduction in file size in the decompressed image.</p>||



|![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.007.png)|![](ReadMe/Aspose.Words.3867ab57-56ef-4d4d-a4b6-5609bb676745.008.png)|
| :- | :- |
|<p>As described above, the image has reduced in size as a result of Run-Length Encoding. In the images above, one can see such a contrast between the original image and the reconstructed image. The left image shows the original image, and the right image shows the reconstructed</p><p>image. The sizes went from 1.17 MB to 900 KB.</p>||


There were several challenges to overcome, including splitting the image into three channels. After all, it differs from a greyscale image in that it does not need to be split into three channels. As a result, working with the three channels was challenging since a greyscale image only needed the jpeg process once, while the colour image required a jpeg algorithm for each channel. To decompress the data, each channel needs to be decompressed and then merged to produce an image.

# <a name="_bookmark5"></a>Conclusion
In modern computing, image compression has become an indispensable part of the workflow. As a result of the ability to compress images to a fraction of their original size, image files can be transmitted faster between people, saving time, disk space, and money. JPEG is an excellent method compared to other compression methods such as PNG. The results achieved even when JPEG is not optimized show the results that can be achieved. We could still consistently achieve 50% or more compression with a lower MSE and a high PSNR. However, more features could be added to this JPEG algorithm in the future. For example, we could implement Huffman encoding in addition to the Run-length Encoding we did. We could also try different ways of doing JPEG compression since it is widely supported.


# <a name="_bookmark6"></a>Acknowledgments
When undergoing this project, we used external sources for guidance to create the compression algorithm. ZigZag scanning logic is implemented in [1], colour separation and manipulation is implemented in [2], and decompression is implemented in [3].


# <a name="_bookmark7"></a>References
1. uk.mathworks.com. (n.d.). Zigzag Scan. [online] Available at: https://uk.mathworks.com/matlabcentral/fileexchange/15317-zigzag-scan [Accessed 5 Dec. 2022].
1. mVirtuoso21 (2022). JPEG-Image-Compressor. [online] GitHub. Available at: https://github.com/mVirtuoso21/JPEG-Image-Compressor/blob/main/main.py [Accessed 5 Dec. 2022].
1. uk.mathworks.com. (n.d.). JPEG Compression (DCT). [online] Available at: https://uk.mathworks.com/matlabcentral/fileexchange/34958-jpeg-compression-dct [Accessed 5 Dec. 2022].
1. GeeksforGeeks. (2020). *Process Of JPEG Data compression*. [online] Available at: https://[www.geeksforgeeks.org/process-of-jpeg-data-compression/.](http://www.geeksforgeeks.org/process-of-jpeg-data-compression/)
1. Dias, D. (2017). *JPEG Compression Algorithm*. [online] Medium. Available at: https://medium.com/breaktheloop/jpeg-compression-algorithm-969af03773da.
1. [www.javatpoint.com.](http://www.javatpoint.com/) (n.d.). *JPEG Compression - Javatpoint*. [online] Available at: https://[www.javatpoint.com/jpeg-compression](http://www.javatpoint.com/jpeg-compression) [Accessed 5 Dec. 2022].
