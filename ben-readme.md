# April Tags: Quad Detection in Images

For this project, we decided to delve into the algorithm that allows computers to detect April Tags in images. As this is a large and complicated algorithm, we decided to focus on two steps: segmentation and quadrilateral (quad) detection.

## Background

April Tags are specially-designed black-and-white squares that are made to be easily converted into position and orientation information.

![April Tag Example](images/tag_36h11.png)

*Example April Tag*

The overarching steps are as follows:

1. Calculate the gradient of the image
2. Group together pixels which have a similar gradient
3. Convert those groups of pixels to line segments
4. Find candidate quadrilaterals from that image
5. Compute the linear transformation from the quadrilateral to the camera
6. Scan the quadrilateral for the light and dark regions to determine if it is a known April Tag, and which Tag it was if so

In this project, we worked our way through steps 1-4 of the April Tag detection process, using the algorithm described [Olson's original April Tag Paper](https://april.eecs.umich.edu/media/pdfs/olson2011tags.pdf) as a point of reference.

## Implementation

###  Pixel Clustering

We started by converting the input image into black-and-white. From here, we used 2-D convolution to approximate the gradient of the image, using the Scharr Operator (convolving a special 3-by-3 matrix and its transpose with the image to calculate the X and Y gradient). From here, we calculated the magnitude and direction of the image gradient.

Once we had the gradient, we grouped pixels which had similar gradients. To do this, we created a graph of the image, where each pixel shares an edge with each of its adjacent neighbors. This edge is assigned a weight equal to the difference in gradient direction between them. We also assigned each pixel to its own "cluster". We then sorted the edges from the least weight to most weight. In order of increasing weight, we take the two clusters the two pixels of that edge belong to, and consider combining them (if they are already in the same cluster, we skip the edge and continue). We combine the two clusters if:

1. The range of gradient directions in the resulting cluster is less than or equal to the range of directions in cluster A or cluster B (whichever is smaller), plus some constant divided by the number of pixels in the resulting cluster. Formally, for clusters $m$ and $n$, we combine them if and only if: $D(n \cup m) \le \text{min}(D(n), D(m)) + \frac{K_D}{|n \cup m|}$, where $D(c)$ gives the range of gradient directions in cluster $c$
2. The range of gradient magnitudes in the resulting cluster is less than or equal to the range of magnitudes in cluster A or cluster B (whichever is smaller), plus some constant divided by the number of pixels in the resulting cluster. Formally, for clusters $m$ and $n$, we combine them if and only if: $M(n \cup m) \le \text{min}(M(n), M(m)) + \frac{K_M}{|n \cup m|}$, where $M(c)$ gives the range of gradient magnitudes in cluster $c$

As recommended by the paper, we used values of 100 and 1200 for $K_D$ and $K_M$, respectively, though they claim that a wide range of values should work.

Repeating this for every edge will result in groups of pixels which follow straight line edges (or are in patches of very consistent color). To convert these groups into lines, we run Principal Component Analysis on the set of points to compute a unit vector pointing in the direction of the best fit line. From here, we project each of the points onto that vector, and find the ones that project furthest in either direction. These projected points are defined as the end points of the line segment for that cluster. Finally, we sort the two points so that travelling along the line from the first point to the second point will keep the darker side on the right to help with quad detection.
