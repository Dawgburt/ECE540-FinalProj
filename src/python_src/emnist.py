import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
from tensorflow.keras import regularizers

# Determine if GPU is usuable, otherwise use CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        # dynamically allocate GPU memory as needed
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

    # Preprocess set: normalize orient one hot encode features
    def preprocess(image, label):
        # covnert 8 bit grayscale value to 0-1 float
        image = tf.cast(image, tf.float32) / 255.0
        # EMNIST set is rotated by default, rate it back to normal
        image = tf.image.transpose(image)
        image = tf.image.flip_left_right(image)
        # Add dimension channel (28, 28, -> 1 <-)
        image = tf.expand_dims(image, -1)
        # turn the label into a one hot encoded array
        label = tf.one_hot(label, num_classes)
        return image, label

    batch_size = 128
    auto_tune   = tf.data.AUTOTUNE

    # Training set
    ds_train = (ds_train
        .map(preprocess, num_parallel_calls=auto_tune) # preprocess data set using multiple cores
        .cache() # save preprosses data
        .shuffle(10_000) # shuffle order training data is fed
        .batch(batch_size)
        .prefetch(auto_tune) # Allow data to be prepared while model is exicuting
    )

    # Validation Set
    ds_test = (ds_test
        .map(preprocess, num_parallel_calls=auto_tune)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    # Build model with Batch normalization, Dropout, L2
    weight_decay = 1e-4
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
            input_shape=(28, 28, 1),
            kernel_regularizer=regularizers.l2(weight_decay)), # prevent large weights and overfitting (L2)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(), # take the max value from the pool
        tf.keras.layers.Dropout(0.25), # set 25% of the neurons to zero 

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)), # prevent large weights (L2)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)), # prevent large weights (L2)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(), 
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation='relu',
                              kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # optimize gradient decent + momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    
    # attach optimizer to model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    # LR Adujustment call back
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau( # reduces learning rate when accuracy plateaus
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    # Early stop call back
    es_cb = tf.keras.callbacks.EarlyStopping( # if accuracy change plateaus for to long stop training
        monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1
    )
    # Call back for saving current weights if they're the bet so far and providing live training stats update
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        'weight_files/best_emnist.weights.h5',    # weights only filename
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Activate the training
    # history = model.fit(
    #     ds_train,
    #     epochs=50,
    #     validation_data=ds_test,
    #     callbacks=[lr_cb, es_cb, ckpt_cb]
    # )

    # Revert to best weights and evaluate to show the final best model performance
    model.load_weights('weight_files/best_emnist.weights.h5')
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print(f"Final Test:\n\tAccuracy: {test_acc:.2%}\n\tloss: {test_loss:.4f}")

    # Save model info in various forms
    print("Saving model info")
    
    # Export the model to a json file
    with open('weight_files/emnist_model.json','w') as f:
        f.write(model.to_json())

    # save weights
    model.save_weights('weight_files/emnist_weights.weights.h5')

    # NumPy archive of weights
    weights_dict = {}
    for layer in model.layers:
        w = layer.get_weights()
        # BatchNorm layers have 4 arrays: [gamma, beta, mean, variance]
        if len(w) == 4:
            weights_dict[f"{layer.name}/gamma"]          = w[0]
            weights_dict[f"{layer.name}/beta"]           = w[1]
            weights_dict[f"{layer.name}/moving_mean"]    = w[2]
            weights_dict[f"{layer.name}/moving_variance"]= w[3]
        # Conv2D and Dense layers have 2 arrays: [kernel, bias]
        elif len(w) == 2:
            weights_dict[f"{layer.name}/kernel"]         = w[0]
            weights_dict[f"{layer.name}/bias"]           = w[1]
    # write out the expanded archive
    np.savez('weight_files/emnist_weights.npz', **weights_dict)


    # save label to character mapping
    label_names = ds_info.features['label'].names
    with open('weight_files/label_names.json','w') as f:
        json.dump(label_names, f, indent=2)

    print("Saving comeplte: emnist_model.json, emnist_weights.npz, label_names.json")
