#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

CSCI-UA 480: Computer Vision
@author: Larry Liu

"""

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
#% matplotlib inline

# insert the location of the images
puzzle_address = 'pics/Convention.jpg'
waldo_address = 'pics/Wally.png'

# load the puzzle and waldo images
puzzle = cv2.imread(puzzle_address)
waldo = cv2.imread(waldo_address)
(waldoHeight, waldoWidth) = waldo.shape[:2]

# display the dimensions and plot the image of Waldo
print("Height of Template: %d & Width of Template: %d" %(waldoHeight, waldoWidth))

# cv2 reads the image in BGR but we need to convert it to RGB
waldo_rgb = cv2.cvtColor(waldo, cv2.COLOR_RGB2BGR)

plt.figure(figsize = (1, 1))
plt.imshow(waldo_rgb)

plt.figure(figsize=(15,15))
puzzle_rgb = cv2.cvtColor(puzzle, cv2.COLOR_RGB2BGR)

plt.imshow(puzzle_rgb)

# find the waldo in the puzzle
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

# grab the bounding box of waldo and extract him from the puzzle image
topLeft = maxLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = puzzle[topLeft[1] : botRight[1], topLeft[0] : botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the puzzle except for Waldo
mask = np.zeros(puzzle.shape, dtype = "uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

puzzle[topLeft[1] : botRight[1], topLeft[0] : botRight[0]] = roi

# display the images
cv2.imwrite("pics/Puzzle_Result.jpg", puzzle)
result_rgb = cv2.cvtColor(puzzle, cv2.COLOR_RGB2BGR)
plt.figure(figsize = (15, 15))
plt.imshow(result_rgb)

# new Waldo Problem
waldo2 = cv2.imread('pics/waldo_books_dim.jpg')
waldoRGB2 = cv2.cvtColor(waldo2, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (3, 3))
plt.imshow(waldoRGB2)

# use zoo image as the large puzzle
puzzle2 = cv2.imread('pics/waldo_zoo.jpg')
puzzleRGB2 = cv2.cvtColor(puzzle2, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (15, 15))
plt.imshow(puzzleRGB2)

# same logic as before
(waldoHeight2, waldoWidth2) = waldo2.shape[:2]
(puzzleHeight, puzzleWidth) = puzzle2.shape[:2]

print("Original Height of Template: %d & Original Width of Template: %d" %(waldoHeight2, waldoWidth2))

# initial scale doesn't matter as it's constantly changing
scale = 0.3
highCorr = float('-inf')
bestHeight, bestWidth, bestScale = 0, 0, scale
finalMaxLoc = None

# use different sizes for the puzzle until the template matches
while scale <= 2.0:
    temp = cv2.resize(puzzle2, (0, 0), fx = scale, fy = scale)
    result2 = cv2.matchTemplate(temp, waldo2, cv2.TM_CCOEFF)
    (_, newCorr, newMinLoc, newMaxLoc) = cv2.minMaxLoc(result2)
    # if there's a high correlation keep track of the value and coordinates
    if newCorr > highCorr:
        highCorr = newCorr
        bestScale = scale
        finalMaxLoc = newMaxLoc
    scale += 0.05

# use the new found best scale for the puzzle to find Waldo
puzzle2 = cv2.resize(puzzle2, (0, 0), fx = bestScale, fy = bestScale)
# map out coordinates of where the high correlation match occurred
topLeft = finalMaxLoc
botRight = (topLeft[0] + waldoWidth2, topLeft[1] + waldoHeight2)
# establish region of interest
roi = puzzle2[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

# mask the rest of the puzzle to highlight Waldo's location
mask = np.zeros(puzzle2.shape, dtype = "uint8")
puzzle2 = cv2.addWeighted(puzzle2, 0.25, mask, 0.75, 0)
puzzle2[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
# create new file to show where he was
cv2.imwrite("pics/found.jpg", puzzle2)
result_rgb = cv2.cvtColor(puzzle2, cv2.COLOR_RGB2BGR)
# also show in console
plt.figure(figsize = (15,15))
plt.imshow(result_rgb)
