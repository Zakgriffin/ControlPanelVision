import cv2
import numpy as np
import math
from cmath import rect, phase

cap = cv2.VideoCapture(0) # video capture

colors = {
    'red': {
        'low': np.array([160, 0, 0]),
        'high': np.array([180, 255, 255]),
        'highlight': (0, 0, 255),
        'angle': 0
    },
    'yellow': {
        'low': np.array([10, 0, 150]),
        'high': np.array([25, 255, 255]),
        'highlight': (0, 255, 255),
        'angle': 45
    },
    'blue': {
        'low': np.array([90, 0, 150]),
        'high': np.array([140, 255, 255]),
        'highlight': (255, 255, 0),
        'angle': 90
    },
    'green': {
        'low': np.array([35, 0, 150]),
        'high': np.array([50, 255, 255]),
        'highlight': (0, 255, 0),
        'angle': 135
    }
}

# finds average angle given list of angles (harder than you might think)
def mean_angle(deg):
    return math.degrees(phase(sum(rect(1, math.radians(d)) for d in deg)/len(deg)))

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'): break # exit when 'q' pressed

    # preprocess
    _, frame = cap.read()
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # find centers to color slices
    centers = []
    for name, color in colors.items():
        mask = cv2.inRange(hsv, color['low'], color['high'])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # look through contours
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3 and cv2.contourArea(approx) > 700:
                cv2.drawContours(frame, [approx], 0, color['highlight'], 2)

                # grab center
                M = cv2.moments(approx)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                centers.append(((cX, cY), name))
                cv2.circle(frame, (cX, cY), 5, color['highlight'], -1)
        """
        if name == 'red':
            cv2.imshow('HSV', mask)
        """

    if len(centers) != 0:
        overallCenter = [0, 0]

        # get overall center of mass
        for p, _ in centers:
            overallCenter[0] += p[0]
            overallCenter[1] += p[1]
        overallCenter[0] //= len(centers)
        overallCenter[1] //= len(centers)
        cv2.circle(frame, tuple(overallCenter), 5, (255, 255, 255), -1)

        # find angles for each color slice center relative to overall center with color offset
        angles = []
        for p, name in centers:
            angle = math.atan2(p[1] - overallCenter[1], p[0] - overallCenter[0])
            angle = (math.degrees(angle) - colors[name]['angle']) * 2
            angles.append(angle)

        # compute overall average angle
        averageAngle = mean_angle(angles) / 2
        if(averageAngle < 0):
            averageAngle += 180

        cv2.putText(frame, str(int(averageAngle)) + ' deg', (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    cv2.imshow('Frame', frame)

# when exited, release the capture
cap.release()
cv2.destroyAllWindows()