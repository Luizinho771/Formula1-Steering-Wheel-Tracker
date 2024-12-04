import numpy as np
import cv2 as cv 


# Define the path to the video and frame file
video_path = './media/v2.mp4'
frame_path = './media/frame.jpg'

    
def showImage(image):
    cv.imshow('window', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getFirstFrame(video_path):
    # Open the video file
    cap = cv.VideoCapture(video_path)
    if(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv.imwrite('./media/frame.jpg', frame)
            #showImage('./media/frame.jpg')
        else:
            print("Error: Could not read frame.")
        cap.release()
    else:
        print("Error: Could not open video file.")
        cap.release()
    
    return frame

def toGrayScale(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Histogram equalization
    equalized = cv.equalizeHist(gray_image)
    # Blur the image to reduce noise
    blurred_frame = cv.GaussianBlur(equalized, (1, 1), 0)
    # Save the images and show them
    cv.imwrite('./media/equalized.jpg', equalized)
    cv.imwrite('./media/gray_frame.jpg', gray_image)
    cv.imwrite('./media/blurred_frame.jpg', blurred_frame)
    #showImage(gray_image)
    #showImage(equalized)
    #showImage(blurred_frame)
    return blurred_frame    

def roi(gray_frame):
    height, width = gray_frame.shape[:2]
    y_start, y_end = int(height * 0.7), int(height * 0.88)  # Limita verticalmente
    x_start, x_end = int(width * 0.45), int(width * 0.7)    # Limita horizontalmente

    # Recorta o frame para a ROI
    roi = gray_frame[y_start:y_end, x_start:x_end]
    
    cv.imwrite('./media/roi.jpg', roi)
    #showImage(roi)
    return roi

def edgeDetection(image):
    edges = cv.Canny(image, threshold1=50, threshold2=150)

    # Aplicar dilatação e fechamento morfológico para preencher as bordas
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(edges, kernel, iterations=1)
    closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel, iterations=2)

    cv.imwrite('./media/edges.jpg', edges)
    cv.imwrite('./media/dilated.jpg', dilated)
    cv.imwrite('./media/closed.jpg', closed)
    #showImage(edges)
    #showImage(dilated)
    #showImage(closed)
    return closed


def findContours(edges, frame):
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Filtrar contornos pela área, mantendo o contorno do volante
    volante_contour = max(contours, key=cv.contourArea)
    # Limites da ROI
    height, width = frame.shape[:2]
    y_start, y_end = int(height * 0.7), int(height * 0.88)  # Limita verticalmente
    x_start, x_end = int(width * 0.45), int(width * 0.7)    # Limita horizontalmente
    volante_contour = volante_contour + [x_start, y_start]

    frame_contours = cv.drawContours(frame.copy(), [volante_contour], -1, (0, 255, 0), 2)
    
    cv.imwrite('./media/frame_contours.jpg', frame_contours)
    #showImage(frame_contours)
    return volante_contour


def steeringWheelPosition(gray_frame, volante_contour):
    mask = cv.drawContours(np.zeros_like(gray_frame), [volante_contour], -1, 255, thickness=cv.FILLED)
    isolated_volante = cv.bitwise_and(gray_frame, gray_frame, mask=mask)
    #showImage(isolated_volante)
    cv.imwrite('./media/isolated_volante.jpg', isolated_volante)
    cv.waitKey(0)
    cv.destroyAllWindows()

def opticalFlow(volante_contour):
    # Initialize video capture
    cap = cv.VideoCapture(video_path)

    # First frame and detection of key points in the contour of the steering wheel
    ret, first_frame = cap.read()
    gray_first = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Mask for the initial contour of the steering wheel
    mask = np.zeros_like(gray_first)
    cv.drawContours(mask, [volante_contour], -1, 255, thickness=cv.FILLED)

    # Initial keypoints within the steering wheel contour
    p0 = cv.goodFeaturesToTrack(gray_first, mask=mask, maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Define the codec and create VideoWriter object to save output video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv.CAP_PROP_FPS)
    out = cv.VideoWriter('./media/output.avi', fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))

    # Loop to process frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv.calcOpticalFlowPyrLK(gray_first, gray_frame, p0, None, **lk_params)

            # Select only the points that were found
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracking points on the frame
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                # Update previous frame and points
                gray_first = gray_frame.copy()
                p0 = good_new.reshape(-1, 1, 2)

            # Reset keypoints if they are lost
            if len(good_new) < 4:
                print(f"Few keypoints left ({len(good_new)}), reinitializing at frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}...")
                p0 = cv.goodFeaturesToTrack(gray_frame, mask=mask, maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
                if p0 is None or len(p0) == 0:
                    print("No valid keypoints found.")
                    #break  # Exit the loop if no keypoints can be found

        else:
            print("No initial keypoints, skipping frame.")
            #break

        # Write the frame to the output video
        out.write(frame)

        # Show the frame with tracking points
        cv.imshow('Steering Wheel Tracking', frame)
        if cv.waitKey(30) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()






if __name__ == "__main__":
    # Get the first frame of the video
    
    first_frame = getFirstFrame(video_path)
    # Convert to grayscale and gaussian blur
    gray_frame = toGrayScale(first_frame)
    # Region of interest
    roi = roi(gray_frame)
    # Edge detection
    edge_frame = edgeDetection(roi)
    
    volante_contour = findContours(edge_frame, first_frame)

    final_frame = steeringWheelPosition(gray_frame, volante_contour)

    opticalFlow(volante_contour)

