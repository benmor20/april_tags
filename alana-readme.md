## Possible Future Improvements

Given more time, our team would have implemented the final two steps of April Tag detection: pose estimation and decoding.

#### Pose Estimation
One common use case for April Tags is visual localization. With knowledge of the real-life dimensions and expected orientation of the April Tag, camera specs, and the cartesian coordinates of the corners of any given quadrilateral found with our program from an image taken by the camera, we could compute a homography transformation from the quadrilateral to the camera. This would be a linear transformation between the plane the quadrilateral is on to the plane the camera is on. This relationship could be used to to estimate the pose of the camera relative to the April Tag's position and orientation in the world.

#### Decoding
Olson's original paper describes building a model of the identified quadrilaterals, assigning each one to be white or black by sampling the outer border of the tag for what values count as white/black in different areas of the image to account for possible lighting variations. A coding system based on lexicodes is used to check if this constructed model matches a valid April Tag or not. This coding system rejects codewords that would create simple patterns likely to occur in the environment (like a single black and white stripe), and accounts for all four possible 90-degree rotations of any given tag by defining the minimum Hamming distance in each case. Lexicodes have been pre-generated for several libraries of April Tags. If a lexicode generated from the model of the identified April Tag candidate has a Hamming distance below a certain threshold (suggested 9 or 10) compared to a valid code word, it is considered to be that April Tag. 

## Lessons Learned for Future Robotic Programming Projects

Our team took a lot of time in the beginning stages of our project to make sense of the paper we were working from ([AprilTag: A robust and flexible visual fiducial system, Olson 2011](https://april.eecs.umich.edu/media/pdfs/olson2011tags.pdf)). We puzzled out what each section of the paper was saying via time and whiteboard space, and wrote detailed pseudo code and docstrings together as a group. This allowed us to efficiently split up work and independently write functions that would be compatible with each other.

We made use of the Liveshare extension in Visual Studio Code for editing and testing of our code. We ran into a few difficulties as we learned that files could not be directly uploaded into a host's folder via the Liveshare and instead had to be uploaded on a team member's end and then pushed to the GitHub repository, but it was overall immensely useful for simultaneous editing of code. It was also great to not all be crowded around one laptop.

There are many nested algorithms and optimization tricks which are worth exploring. Drawing on existing knowledge in order to understand how these algorithms were designed and where optimization can occur will be immensely useful in future projects both for familiarity with specific algorithms as well as intuition about computational efficiency and where it might be improved.