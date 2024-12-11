import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display_camera/display_photo/test")
mode = ap.parse_args().mode

train_dir = 'data/train'
val_dir = 'data/test'

batch_size = 64
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])

def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axs[0].set_title('Dokładność modelu')
    axs[0].set_xlabel('Epoki')
    axs[0].set_ylabel('Dokładność')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Train Loss', color='blue')
    axs[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axs[1].set_title('Strata modelu')
    axs[1].set_xlabel('Epoki')
    axs[1].set_ylabel('Strata')
    axs[1].legend()

    plt.savefig('model_training_history.png')
    plt.show()

emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

def detect_emotion_from_image(image_path):
    model.load_weights('model_weights.weights.h5')

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Nie znaleziono obrazu pod podaną ścieżką.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48, 48))
    cropped_img = np.expand_dims(np.expand_dims(gray_resized, -1), 0) / 255.0

    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    confidence = float(prediction[0][maxindex])


    return emotion_dict[maxindex], cv2.cvtColor(image, cv2.COLOR_BGR2RGB), confidence

global_cap = None

def initialize_camera():
    global global_cap
    if global_cap is None:
        global_cap = cv2.VideoCapture(0)
        if not global_cap.isOpened():
            raise RuntimeError("Nie można uzyskać dostępu do kamery.")

def release_camera():
    global global_cap
    if global_cap is not None:
        global_cap.release()
        global_cap = None

def detect_emotion_from_camera():
    global global_cap
    if global_cap is None:
        raise RuntimeError("Kamera nie jest zainicjowana. Wywołaj initialize_camera najpierw.")

    emotion = "Neutral"
    ret, frame = global_cap.read()
    if not ret:
        raise RuntimeError("Nie udało się odczytać obrazu z kamery.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) / 255.0
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, emotion

if mode == "test":
    model.load_weights('model_weights.weights.h5')

    y_true = validation_generator.classes
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Macierz pomyłek:")
    print(cm)

    report = classification_report(y_true, y_pred_classes, target_names=list(emotion_dict.values()))
    print("\nRaport klasyfikacji:")
    print(report)

elif mode == "train":
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )

    model.save_weights('model_weights.weights.h5')

    plot_model_history(history)

elif mode == "display_camera":
    initialize_camera()
    try:
        while True:
            frame, emotion = detect_emotion_from_camera()
            cv2.putText(frame, emotion, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Emotion Detection', cv2.resize(frame, (800, 600)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        release_camera()
        cv2.destroyAllWindows()

elif mode == "display_photo":
    image_path = 'photos/neutral.png'
    emotion, image = detect_emotion_from_image(image_path)
    print(f"Emotion predicitons: {emotion}")
    cv2.imshow('Emotion Detection', cv2.resize(image, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Undefined mode. Use --mode train/display_camera/display_photo/test.")
