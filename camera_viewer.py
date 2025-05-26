import cv2
import numpy as np

def initCamera(
    camera=0,
    width=1280,
    height=960,
    fps=120,
    exposure=1,
    gain=1,
    gamma=72,
    contrast=32,
):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_GAMMA, gamma)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    return cap

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_point is None:  # First click
            start_point = (x, y)
            end_point = None
        else:  # Second click
            end_point = (x, y)

def main():
    global start_point, end_point, drawing
    start_point = None
    end_point = None
    drawing = False
    
    cap = initCamera()
    cv2.namedWindow('Camera Viewer')
    cv2.setMouseCallback('Camera Viewer', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy of the frame for drawing
        display = frame.copy()
        
        # Draw the first point if it exists
        if start_point:
            cv2.circle(display, start_point, 5, (0, 0, 255), -1)  # Red dot for first point
        
        # Draw the line if we have both points
        if start_point and end_point:
            cv2.line(display, start_point, end_point, (0, 255, 0), 2)
            # Calculate pixel distance
            distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            # Display distance
            cv2.putText(display, f'Distance: {distance:.1f} pixels', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Viewer', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('c'):  # Clear the line
            start_point = None
            end_point = None
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 