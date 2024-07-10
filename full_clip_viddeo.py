import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime, timedelta
import csv
from win10toast import ToastNotifier
import smtplib
from email.mime.text import MIMEText
import winsound

# Load the YOLOv8 model
model = YOLO('best.pt')

# Booleans
alaram=True
send_email=False

# Define video parameters
output_width, output_height = 640, 360
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')
current_date_time = datetime.now()


# Define function to create video writer
def create_video_writer(output_path):
    return cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

# URL of the IP web camera stream
web_camera_url = (0)

# Open the IP web camera stream
web_camera = cv2.VideoCapture(web_camera_url)

# Create folder to save the video
output_folder_full = "full_videos"
output_folder_clip = "clipped_videos"
os.makedirs(output_folder_full, exist_ok=True)
os.makedirs(output_folder_clip, exist_ok=True)

# CSV file to store gun detection instances
csv_file_path = "Alerts.csv"
with open(csv_file_path, 'w+', newline='') as csvfile:
    fieldnames = ['alert_type', 'Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Create a video writer for the entire video stream
    video_output_path_full = os.path.join(output_folder_full, f"gun_detection_output_full{current_date_time.strftime('%Y_%m_%d,%H_%M_%S')}.avi")
    video_writer_full = create_video_writer(video_output_path_full)

    video_writer = None

    # Email settings
    email_subject = "Gun Detection Alert"
    email_body = "A gun has been detected in the video stream."
    email_from = "umarrai024@gmail.com"
    email_to = "umarrai024@gmail.com"
    email_password = "pegp qxwu klqb bsmp"

    # while datetime.now() < end_time:
    while True:    
        found_objects = None
        gun_found=False
        success, frame = web_camera.read()
        if not success:
            break

        # Perform gun detection using YOLOv8 model
        result = model(frame)
        for data in result:
            print(data.names)
            found_objects = data.names
            if 'gun' in str(found_objects):
                try:
                    print("Gun probability:", data.probs[0])
                except:
                    pass
                gun_found = True
            

        # Check if gun is detected
        if len(result[0]) > 0 and gun_found:

            # Alaram
            if alaram:
                winsound.Beep(2500, 1000)
            # Get current date and time
            detection_time = datetime.now()

            # Write the event and detection time to CSV
            writer.writerow({'alert_type': 'Gun Detection', 'Time': detection_time.strftime("%Y-%m-%d %H:%M:%S")})
            print("Gun detected at:", detection_time.strftime("%Y-%m-%d %H:%M:%S"))

            # Send email
            if send_email:
                msg = MIMEText(email_body)
                msg['Subject'] = email_subject
                msg['From'] = email_from
                msg['To'] = email_to

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(email_from, email_password)
                server.sendmail(email_from, email_to, msg.as_string())
                server.quit()

            # toaster = ToastNotifier()
            # toaster.show_toast("Gun Detected!", "A gun has been detected in the video stream", duration=5)
            

            # Create a new video writer for the detected event
            if video_writer is not None:
                video_writer.release()  # Release the previous video writer

            video_output_path = os.path.join(output_folder_clip, f"gun_detection_output_{detection_time.strftime('%Y_%m_%d,%H_%M_%S')}.avi")
            video_writer = create_video_writer(video_output_path)

        # Visualize the results on the frame
        annotated_frame = result[0].plot() if len(result) > 0 else frame

        # Resize the frame to fit the output dimensions
        annotated_frame_resized = cv2.resize(annotated_frame, (output_width, output_height))

        # Write the frame to the video file
        if video_writer is not None:
            video_writer.write(annotated_frame_resized)

        # Write the frame to the full video file
        video_writer_full.write(annotated_frame_resized)

        if len(result) > 0:
            # Display the frame
            cv2.imshow("Gun Detection - YOLOv8 Inference", annotated_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
web_camera.release()
if video_writer is not None:
    video_writer.release()
if video_writer_full is not None:
    video_writer_full.release()
cv2.destroyAllWindows()