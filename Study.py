import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, random
from sklearn.metrics import confusion_matrix, classification_report

base_path = './dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))
batch_size_value = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 225, shear_range=0.1, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
    vertical_flip=True, validation_split=0.1)

test_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=batch_size_value,
    class_mode='categorical', subset='training', seed=0)

validation_generator = test_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=batch_size_value,
    class_mode='categorical', subset='validation', seed=0)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(4, activation='softmax')
])

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('./results/learning_rate.png')

history = LossHistory()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit_generator(train_generator, epochs=150, steps_per_epoch=1789 // 32, validation_data=validation_generator,
                    validation_steps=198 // 32, callbacks=[history])

model.save('./results/my_model.h5')

# Load the model
model = load_model('./results/my_model.h5')

# Define base path and load image paths
base_path = './dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

# Set batch size and image dimensions
batch_size_value = 16
img_width, img_height = 300, 300

# Create data generator for the test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    base_path,
    target_size=(img_width, img_height),
    batch_size=batch_size_value,
    class_mode='categorical',
    shuffle=False
)

# Make predictions on the test data
y_pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
true_classes = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(true_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Compute classification report
report = classification_report(true_classes, y_pred_classes, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(report)

# Compute accuracy, precision, and recall
accuracy = np.sum(true_classes == y_pred_classes) / len(true_classes)
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Plot the loss function curve
plt.figure()
plt.plot(history.losses['epoch'], label='Training Loss')
plt.plot(history.val_loss['epoch'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.legend()
plt.savefig('./results/loss_function.png')

# Plot the accuracy curve
plt.figure()
plt.plot(history.accuracy['epoch'], label='Training Accuracy')
plt.plot(history.val_acc['epoch'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig('./results/accuracy.png')

# Plot precision and recall curves
plt.figure()
plt.plot(recall, label='Recall')
plt.plot(precision, label='Precision')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision and Recall')
plt.legend()
plt.savefig('./results/precision_recall.png')

# Save the classification report to a text file
with open('./results/classification_report.txt', 'w') as f:
    f.write(report)
