import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
from tensorflow.keras import regularizers
from PIL import Image  # for local image loading

# Determine if GPU is usable, otherwise use CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus[0].name}")
    DEVICE = '/GPU:0'
else:
    print("No GPU found, using CPU")
    DEVICE = '/CPU:0'

with tf.device(DEVICE):
    # Load EMNIST balanced set
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/balanced',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    num_classes = ds_info.features['label'].num_classes

    # Preprocess set: normalize, orient, one-hot encode
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.transpose(image)
        image = tf.image.flip_left_right(image)
        image = tf.expand_dims(image, -1)
        label = tf.one_hot(label, num_classes)
        return image, label

    batch_size = 128
    auto_tune = tf.data.AUTOTUNE

    ds_train = (
        ds_train
        .map(preprocess, num_parallel_calls=auto_tune)
        .cache()
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(auto_tune)
    )
    ds_test = (
        ds_test
        .map(preprocess, num_parallel_calls=auto_tune)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    # Build model
    weight_decay = 1e-4
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
            input_shape=(28, 28, 1),
            kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1
    )
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        'weight_files/best_emnist.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # (Uncomment to train)
    # history = model.fit(
    #     ds_train,
    #     epochs=10,
    #     validation_data=ds_test,
    #     callbacks=[lr_cb, es_cb, ckpt_cb]
    # )

    # Evaluate on test set
    model.load_weights('weight_files/best_emnist.weights.h5')
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print(f"Final Test:\n\tAccuracy: {test_acc:.2%}\n\tLoss: {test_loss:.4f}")

    # Save model architecture + weights
    print("Saving model info")
    with open('weight_files/emnist_model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('weight_files/emnist_weights.weights.h5')

    # Save expanded weights archive
    weights_dict = {}
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) == 4:
            weights_dict[f"{layer.name}/gamma"] = w[0]
            weights_dict[f"{layer.name}/beta"] = w[1]
            weights_dict[f"{layer.name}/moving_mean"] = w[2]
            weights_dict[f"{layer.name}/moving_variance"] = w[3]
        elif len(w) == 2:
            weights_dict[f"{layer.name}/kernel"] = w[0]
            weights_dict[f"{layer.name}/bias"] = w[1]
    np.savez('weight_files/emnist_weights.npz', **weights_dict)

    # Save TFDS label mapping
    label_names = ds_info.features['label'].names
    with open('weight_files/label_names.json', 'w') as f:
        json.dump(label_names, f, indent=2)
    print("Saved: emnist_model.json, emnist_weights.npz, label_names.json")

    # ——————————————————————————————
    # 6) Inference on a local image
    # ——————————————————————————————
    def preprocess_local(path):
        img = Image.open(path).convert('L')
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((28, 28), resample=resample)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr)
        arr = np.fliplr(arr)
        return arr.reshape((1, 28, 28, 1))

    local_path = 'test_imgs/y.jpg'  # ← set your local image path
    x_local = preprocess_local(local_path)
    probs_local = model.predict(x_local, verbose=0)
    idx_local = int(np.argmax(probs_local, axis=1)[0])
    char_local = label_names[idx_local]
    conf_local = float(probs_local[0, idx_local])

    print(f"\nLocal inference on '{local_path}':")
    print(f"\tPredicted index: {idx_local}")
    print(f"\tCharacter label: '{char_local}'")
    print(f"\tConfidence:      {conf_local:.3f}")
