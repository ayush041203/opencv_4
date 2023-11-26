import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)  # You can change the argument to the webcam index if necessary

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize paddle and puck positions
paddle_position = [320, 480]  # Initial position of the paddle
paddle_radius = 30  # Radius of the paddle
puck_position = [320, 240]  # Initial position of the puck

# Initial velocity
initial_puck_velocity = [10, 10]
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
target_image = cv2.imread("target.png", cv2.IMREAD_UNCHANGED)
if target_image is None:
    print("Error: Unable to load the target image.")
else:
    target_image = cv2.resize(target_image, (30, 30))


# Initialize 5 target positions randomly (remember assignment 2!!)
target_positions = np.random.randint(50, 600, size=(5, 2))

# Initialize score
score = 0

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds

# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=5):
    distance = np.linalg.norm(np.array(puck) - np.array(target))
    acceptance_radius = (acceptance_percent / 100) * (target_image.shape[0] / 2)
    return distance <= acceptance_radius


def reset_game():
    global puck_position, puck_velocity, target_positions, score, start_time
    puck_position = [320, 240]
    puck_velocity = initial_puck_velocity.copy()
    target_positions = np.random.randint(50, 600, size=(5, 2))
    score = 0
    start_time = time.time()

reset_game()

while True:
    # Calculate remaining time and elapsed time in minutes and seconds
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - elapsed_time)
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe hands
    result = hands.process(rgb_frame)

    # Update paddle position based on index finger tip
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        paddle_position = [int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])]

    # Update puck position based on its velocity
    puck_position[0] += puck_velocity[0]
    puck_position[1] += puck_velocity[1]

    # Check for collisions with the walls
    if puck_position[0] < 0 or puck_position[0] > frame.shape[1]:
        puck_velocity[0] *= -1
    if puck_position[1] < 0 or puck_position[1] > frame.shape[0]:
        puck_velocity[1] *= -1

    # Check for collisions with the paddle
    paddle_distance = np.linalg.norm(np.array(puck_position) - np.array(paddle_position))
    if paddle_distance < paddle_radius:
        puck_velocity[0] *= -1
        puck_velocity[1] *= -1

    # Check for collisions with the targets (use is_within_acceptance)
    for i, target_position in enumerate(target_positions):
        if is_within_acceptance(puck_position, target_position):
            print(f"Puck hit target {i + 1}")
            score += 1
            target_positions = np.delete(target_positions, i, axis=0)
            puck_velocity = [v + 2 for v in puck_velocity]

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    cv2.circle(frame, (paddle_position[0], paddle_position[1]), paddle_radius, (255, 0, 0), -1)
    cv2.circle(frame, (puck_position[0], puck_position[1]), 10, (0, 0, 255), -1)
    for target_position in target_positions:
        y, y_end = target_position[1], target_position[1] + target_image.shape[0]
        x, x_end = target_position[0], target_position[0] + target_image.shape[1]

        # Perform bounds checking to ensure indices are within the frame
        if 0 <= y < frame.shape[0] and 0 <= y_end < frame.shape[0] and 0 <= x < frame.shape[1] and 0 <= x_end < frame.shape[1]:
            target_roi = frame[y:y_end, x:x_end]
            alpha = target_image[:, :, 3] / 255.0
            beta = 1.0 - alpha
            for c in range(0, 3):
                target_roi[:, :, c] = (alpha * target_image[:, :, c] + beta * target_roi[:, :, c])

    # Display the player's score on the frame
    cv2.putText(frame, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the remaining time on the frame
    cv2.putText(frame, f"Time: {minutes:02}:{seconds:02}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  # Check if all targets are hit or time is up
    if len(target_positions) == 0 or remaining_time <= 0:
        break

    # Display the resulting frame
    cv2.imshow("Game", frame)

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()