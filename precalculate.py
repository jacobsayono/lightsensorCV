import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define bounding box
BOX_SIZE = 50
boxLeft = 875
boxTop = 300

# Load video
cap = cv2.VideoCapture('greenfilter.mp4')

# Get video FPS for time calculation
fps = cap.get(cv2.CAP_PROP_FPS)

# Start and end time in seconds
start_time = 22
end_time = 33

# Calculate start and end frame numbers
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Initialize list to store intensity values
light_intensity = []

frame_num = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if start_frame <= frame_num <= end_frame:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate light intensity in the bounding box
            roi = gray[boxTop:boxTop+BOX_SIZE, boxLeft:boxLeft+BOX_SIZE]
            intensity = np.mean(roi)
            light_intensity.append(intensity)

            # Draw bounding box
            cv2.rectangle(frame, (boxLeft, boxTop), (boxLeft + BOX_SIZE, boxTop + BOX_SIZE), (0, 0, 255), 2)
            
        frame_num += 1
    else:
        break

cap.release()

# Plot light intensity
plt.figure()
plt.plot(np.linspace(start_time, end_time, len(light_intensity)), light_intensity)
plt.xlabel('Time (s)')
plt.ylabel('Light Intensity')
plt.show()
