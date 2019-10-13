#!/usr/bin/python

import sys

import cv2
import numpy as np


def crop_to_interest(img):
    """
    Crop given image to just the points of interests
    """
    h = img.shape[0]
    w = img.shape[1]
    triangle = np.array(
        [[(int(0.15 * w), h), (int(0.85 * w), h), (int(0.5 * w), int(0.4 * h))]])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def filter_colors_hsv(img):
    """
    Convert given image to HSV color space and filter out all the non white and yellow-orangish colors
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_light = np.array([np.round(40 / 2), np.round(0.1 * 255), np.round(0.7 * 255)])
    yellow_dark = np.array([np.round(55 / 2), np.round(1 * 255), np.round(1 * 255)])
    yellow_range = cv2.inRange(img, yellow_light, yellow_dark)

    orange_light = np.array([np.round(20 / 2), np.round(0.5 * 255), np.round(0.6 * 255)])
    orange_dark = np.array([np.round(35 / 2), np.round(0.9 * 255), np.round(0.8 * 255)])
    orange_range = cv2.inRange(img, orange_light, orange_dark)

    white_light = np.array([np.round(0 / 2), np.round(0.0 * 255), np.round(0.7 * 255)])
    white_dark = np.array([np.round(360 / 2), np.round(0.1 * 255), np.round(1 * 255)])
    white_range = cv2.inRange(img, white_light, white_dark)

    masked = cv2.bitwise_and(img, img, mask=(yellow_range | orange_range | white_range))

    return cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)


def update_buffer(line, buffer):
    """
    add given line to a buffer of max size 10
    any older line will be pushed out
    """
    buffer.append(line)
    return buffer[-10:]


def get_line_segment(x1, x2, line):
    """
    Use given slope and y-intercept values of the line to derive
    the y1,y2 values for the inputs x1,x2
    """
    fx = np.poly1d(line)
    return (x1, int(fx(x1))), (x2, int(fx(x2)))


def get_line_intersection(left_line, right_line):
    """
    find the intersection between 2 given lines
    """
    left_slope, left_intercept = left_line
    right_slope, right_intercept = right_line

    # put the coordinates into homogeneous form
    a = [[left_slope, -1], [right_slope, -1]]
    b = [-left_intercept, -right_intercept]
    return np.linalg.solve(a, b)


def partition_line_segments(line_segments, mid_x):
    """
    Partition line segments by their position in the image to determine which is the
    left line vs. the right line. Filter out line segments with slopes outside a
    given minimum / maximum
    """
    left_points = {'X': [], 'Y': [], }
    right_points = {'X': [], 'Y': [], }
    for segment in line_segments:
        x1, y1, x2, y2 = segment[0]
        dY = y2 - y1
        dX = x2 - x1
        if dX != 0:  # don't divide by zero
            slope = float(dY) / float(dX)
            if x1 < mid_x and x2 < mid_x:  # left lines
                if -0.9 < slope < -0.5:
                    left_points['X'] += [x1, x2]
                    left_points['Y'] += [y1, y2]
            elif x1 > mid_x and x2 > mid_x:  # right lines
                if 0.9 > slope > 0.5:
                    right_points['X'] += [x1, x2]
                    right_points['Y'] += [y1, y2]
    return left_points, right_points


def fit_lines_to_points(left_points, right_points, left_buffer, right_buffer):
    """
    fit a line (slope, y-intercept) to each left points sets and right points set and add to buffer
    return the average slope and y-intercept values over the last 10 frames
    """
    if len(left_points['X']) > 1:
        left_line = np.polyfit(left_points['X'], left_points['Y'], 1)
        left_buffer = update_buffer(left_line, left_buffer)
    if len(right_points['X']) > 1:
        right_line = np.polyfit(right_points['X'], right_points['Y'], 1)
        right_buffer = update_buffer(right_line, right_buffer)

    if len(left_buffer) > 0 and len(right_buffer) > 0:
        return np.mean(left_buffer, axis=0), np.mean(right_buffer, axis=0)
    elif len(left_buffer) > 0:
        return np.mean(left_buffer, axis=0), None
    elif len(right_buffer) > 0:
        return None, np.mean(right_buffer, axis=0)
    else:
        return None, None


def get_lane_lines(img, left_buffer, right_buffer):
    """
    find lanes of given image with the help of Hough Line Transform Algorithm
    returns a list of lanes, which could be both left and right lanes, just the left lane, just the right lane or both
    """

    width = img.shape[1]

    line_segments = cv2.HoughLinesP(img, 2, (np.pi / 180), 50, minLineLength=15, maxLineGap=10)

    if line_segments is not None:

        left_points, right_points = partition_line_segments(line_segments, int(width / 2))
        left_line, right_line = fit_lines_to_points(left_points, right_points, left_buffer, right_buffer)

        if left_line is not None and right_line is not None:
            int_x, _ = get_line_intersection(left_line, right_line)

            return [get_line_segment(0, int(int_x), left_line), get_line_segment(width, int(int_x), right_line)]
        elif left_line is not None:
            return [get_line_segment(0, int(width / 2), left_line)]
        elif right_line is not None:
            return [get_line_segment(width, int(width / 2), right_line)]
        else:
            return []
    else:
        return []


def stop():
    cv2.destroyAllWindows()
    exit()


def draw_lanes_overlay(img, left_lane_buffer, right_lane_buffer):
    """
    detect the left and right lane in the given image
    returns the original image with the lanes drawn as an overlay
    """

    color_filtered = filter_colors_hsv(img)

    blurred_img = cv2.GaussianBlur(color_filtered[:, :, 2], (3, 3), 0)

    canny_img = cv2.Canny(blurred_img, 20, 60)

    cropped_img = crop_to_interest(canny_img)

    lane_lines = get_lane_lines(cropped_img, left_lane_buffer, right_lane_buffer)

    line_img = np.zeros_like(img)

    for line in lane_lines:
        cv2.line(line_img, line[0], line[1], [0, 0, 255], 5)

    return cv2.addWeighted(img, 0.8, line_img, 1, 1)


def detect_lanes(road_video_src):
    """
    read from given road video, detect driving lanes and mark them by drawing an overlay on top
    """
    cap = cv2.VideoCapture(road_video_src)

    left_buffer = []
    right_buffer = []

    while cap.isOpened():

        _, frame = cap.read()

        if frame is None:
            break

        cv2.imshow('Lane Detection', draw_lanes_overlay(frame, left_buffer, right_buffer))

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            stop()

    cap.release()


def main():
    src = sys.argv[1] if len(sys.argv) >= 2 else "road-1.mp4"

    while True:
        detect_lanes(src)


if __name__ == "__main__":
    main()
