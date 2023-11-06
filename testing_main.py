def main():
    # Load in the picture
    color_matrix = cv2.imread('images/black_rectangle.png')

    # Call the april tag detector
    quads = april_tag_detector(color_matrix)

    # segments = np.array([[1, 0, 437, 0],
    #                      [381, 105, 59, 105],
    #                      [58, 112, 58, 221],
    #                      [381, 163, 58, 164],
    #                      [382, 221, 382, 107],
    #                      [383, 221, 383, 109],
    #                      [384, 221, 384, 108],
    #                      [59, 222, 381, 222],
    #                      [0, 326, 0, 1],
    #                      [0, 163, 437, 163],
    #                      [438, 1, 438, 326],
    #                      [437, 327, 1, 327]])

    segments = np.array([[0, 0, 0, 1],
                         [0, 1, 1, 1],
                         [1, 1, 1, 0],
                         [1, 0, 0, 0],
                         [-5, -5, -5, 5],
                         [-5, 5, 5, 5],
                         [5, 5, 5, -5],
                         [5, -5, -5, -5]])
    

    print(segments.shape)
    quads_by_idx = quad_detector(segments)
    print(quads_by_idx)
    print(quads_by_idx.shape)

    for quad in quads_by_idx:
        points = np.array([[segments[quad[0], 2], segments[quad[0], 3]],
                            [segments[quad[0], 0], segments[quad[0], 1]],
                            [segments[quad[1], 2], segments[quad[1], 3]],
                            [segments[quad[1], 0], segments[quad[1], 1]],
                            [segments[quad[2], 2], segments[quad[2], 3]],
                            [segments[quad[2], 0], segments[quad[2], 1]],
                            [segments[quad[3], 2], segments[quad[3], 3]],
                            [segments[quad[3], 0], segments[quad[3], 1]]])
        plt.plot(points[:, 0], points[:, 1])
    plt.show()
