import tensorflow as tf

def create_model(sample_length: int) -> tf.keras.Model:
    """
    Create a keras model for predicting noise probabilities

    Parameters
    ----------
    sample_length: int
        Length of each train sample (in bits)

    Returns
    -------
    model
        Keras model
        input shape (None, sample_length, 2)
        output shape (None, sample_length)
    """
    inputs = tf.keras.layers.Input(shape=(sample_length, 2), dtype=tf.float32)
    out = tf.keras.layers.Lambda(lambda x: x * 10)(inputs)  # Basic scaling
    out = tf.keras.layers.Conv1D(16, 5, padding='same', activation='relu')(out)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(out)
    out = tf.keras.layers.Reshape((sample_length,))(out)
    return tf.keras.Model(inputs=inputs, outputs=out)
