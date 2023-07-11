import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define bounding box
BOX_SIZE = 15
boxLeft = 870
boxTop = 568

def updateBoundingBoxPosition(x, y, width, height):
    global boxLeft, boxTop
    boundingBoxWidth = BOX_SIZE
    boundingBoxHeight = BOX_SIZE

    boundingBoxX = max(0, min(x - boundingBoxWidth / 2, width - boundingBoxWidth))
    boundingBoxY = max(0, min(y - boundingBoxHeight / 2, height - boundingBoxHeight))

    # Update the boxLeft and boxTop variables based on the new position and size
    boxLeft = int(boundingBoxX)
    boxTop = int(boundingBoxY)

# Load video
cap = cv2.VideoCapture('nofilter.mp4')

# Get video FPS and total frames for time calculation
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Start and end time in seconds
start_time = 7
end_time = 23

# Calculate start and end frame numbers
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Initialize list to store intensity values
light_intensity = []

# Initialize plt figure
plt.ion()
fig, ax = plt.subplots()

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
            cv2.imshow('Video', frame)
            
            # Plot intensity in real-time
            ax.clear()
            ax.plot(light_intensity, label='Intensity')
            ax.legend(loc='upper left')
            plt.draw()
            plt.pause(0.001)
            
            # Update the bounding box position if needed
            # updateBoundingBoxPosition(new_x, new_y, frame.shape[1], frame.shape[0])
            
            # Quit video by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_num += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
