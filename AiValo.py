import cv2
import numpy as np
import pyautogui
import pytesseract
import tensorflow as tf
import glob

class Model:
    def __init__(self):
        self.reward_memory = []
        
    def reward(self, reward):
        self.reward_memory.append(reward)

    def predict(self, state):
        # Preprocess the state
        state = cv2.resize(state, (256, 256))
        state = state.astype("float32") / 255.0
        # Perform the prediction
        action = trained_model.predict(np.expand_dims(state, axis=0))[0]

        # Convert the output to the desired format
        mouse_movement = (int(action[0]), int(action[1]))
        key_press = chr(int(action[2]))
        left_click = bool(action[3])

        return (mouse_movement, key_press, left_click)

def get_game_state():
    # Get the current screen image
    image = np.array(pyautogui.screenshot())

    # Perform any image processing or feature extraction here

    return image

def send_action(action):
    # Unpack the action tuple
    mouse_movement, key_press , left_click = action

    # Move the mouse to the specified coordinates
    pyautogui.moveTo(*mouse_movement)

    # Press the specified keys
    pyautogui.press(key_press)

    # Perform a left click if specified
    if left_click:
        pyautogui.click()

# Load the training data
X_train = np.array([cv2.imread(f) for f in glob.glob("train/*.jpg")])
y_train = np.array([int(f.split("/")[-1].split(".")[0]) for f in glob.glob("train/*.jpg")])

# Preprocess the data
X_train = X_train.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 3)

# Define the coordinates of the numbers to read
reward_coordinates = (100, 100)
punish_coordinates = (200, 200)

# Define the neural network
trained_model = tf.keras.Sequential()
trained_model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
trained_model.add(tf.keras.layers.Activation('relu'))
trained_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
trained_model.add(tf.keras.layers.Flatten())
trained_model.add(tf.keras.layers.Dense(64))
trained_model.add(tf.keras.layers.Activation('relu'))
trained_model.add(tf.keras.layers.Dense(3))

# Compile the model
trained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights
trained_model.load_weights("weights.h5")

# Use the trained model to play the game
previous_reward_number = 0
previous_punish_number = 0
while True:
    # Get the game state
    state = get_game_state()

    # Make a prediction
    action = Model().predict(state)

    # Send the action to the game
    send_action(action)

    # Take a screenshot of the game
    screenshot = pyautogui.screenshot()

    # Use OCR to read the number at the reward coordinates
    reward_number = pytesseract.image_to_string(screenshot.crop(reward_coordinates + (10, 10)))

    # Use OCR to read the number at the punish coordinates
    punish_number = pytesseract.image_to_string(screenshot.crop(punish_coordinates + (10, 10)))

    # Convert the numbers to integers
    reward_number = int(reward_number) if reward_number.isdigit() else 0
    punish_number = int(punish_number) if punish_number.isdigit() else 0

    # Compare the numbers to the previous iteration
    if reward_number > previous_reward_number:
        # Give a positive reward
        Model().reward(1)
    elif punish_number > previous_punish_number:
        # Give a negative reward
        Model().reward(-1)

    # Update the previous numbers
    previous_reward_number = reward_number
    previous_punish_number = punish_number
