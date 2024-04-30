from sklearn.decomposition import IncrementalPCA
from fcmeans import FCM
# Function to preprocess and extract features from an image
def preprocess_and_extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    return feature.flatten()

# Function to predict cluster label for a single image
def predict_cluster_label(img_path, pca, fcm):
    # Preprocess and extract features from the image
    feature = preprocess_and_extract_features(img_path)
    # Apply PCA transformation if necessary
    feature_pca = pca.transform([feature])  # Assuming 'pca' is the trained PCA object
    # Predict cluster label using Fuzzy C-means
    cluster_label = fcm.predict(feature_pca)  # Assuming 'fcm' is the trained FCM object
    return cluster_label
history = model.fit(X_train,y_train,
                    epochs=10,
                    validation_data=(X_test,y_test),
                    verbose = 1,
                    initial_epoch=0)

%matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

model.save("FCAttentionModel.h5")
