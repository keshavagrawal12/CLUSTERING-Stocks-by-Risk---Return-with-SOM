"""
Dense autoencoder that compresses the 7 features into a 3-D latent code.
Syllabus: Unit 5 - "Autoencoders" (Practicals 13-14)
          Unit 4 - regularisation, dropout, initialisation, gradient-descent strategies.

Architecture:   7 -> 16 -> [3] -> 16 -> 7
  * encoder 7->16->3, decoder 3->16->7; the network learns to reproduce its own input.
  * Self-supervised: the "label" is the input itself, loss = MSE(x, x_hat).
  * Because the middle layer is only 3 wide, the network must squeeze the 7 features into
    3 numbers that keep as much information as possible -> those 3 numbers can then be
    fed to the SOM instead of the raw features.

Unit 4 concepts, each one visible below:
  * He initialisation   - weights scaled for ReLU so activations neither vanish nor explode
  * Dropout             - randomly zeroes hidden units in training -> less co-adaptation
  * L2 regularisation   - penalises large weights (weight decay) -> smoother mapping
  * Adam optimiser      - gradient descent with momentum + per-weight adaptive step sizes
  * Early stopping      - stop when validation loss stops improving, keep the best weights

TensorFlow is optional: if it isn't installed, AVAILABLE is False and the API reports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app import config

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silence TF's start-up chatter

try:  # pragma: no cover - depends on the environment
    import tensorflow as tf
    from tensorflow import keras
    AVAILABLE = True
    IMPORT_ERROR = ""
except Exception as exc:  # ImportError, or a broken install
    tf = keras = None
    AVAILABLE = False
    IMPORT_ERROR = str(exc)


@dataclass
class AutoencoderResult:
    latent: np.ndarray                  # (N, 3) codes - the new SOM input
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    epochs_run: int
    reconstruction_mse: float           # on all stocks, with the best weights
    config: dict
    # decoder(latent codes) -> standardised 7 features. Lets us show the component planes
    # of a latent-space SOM back in real units. Not serialised.
    decode: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)


def train_autoencoder(X: np.ndarray, hidden: int = config.AE_HIDDEN, latent: int = config.AE_LATENT,
                      dropout: float = config.AE_DROPOUT, l2: float = config.AE_L2,
                      learning_rate: float = config.AE_LEARNING_RATE, epochs: int = config.AE_EPOCHS,
                      batch_size: int = config.AE_BATCH_SIZE, val_split: float = config.AE_VAL_SPLIT,
                      patience: int = config.AE_PATIENCE, seed: int = config.AE_SEED) -> AutoencoderResult:
    if not AVAILABLE:
        raise RuntimeError(f"TensorFlow is not installed: {IMPORT_ERROR}")

    keras.utils.set_random_seed(seed)                    # numpy + python + TF seeds together
    dim = X.shape[1]
    he = keras.initializers.HeNormal(seed=seed)          # Unit 4: He initialisation for ReLU
    reg = keras.regularizers.L2(l2)                      # Unit 4: L2 weight penalty

    inputs = keras.Input(shape=(dim,), name="features")
    h = keras.layers.Dense(hidden, activation="relu", kernel_initializer=he,
                           kernel_regularizer=reg, name="enc_hidden")(inputs)
    h = keras.layers.Dropout(dropout, name="enc_dropout")(h)       # Unit 4: dropout
    # Bottleneck: linear so the latent code can take any sign/scale (like PCA scores).
    z = keras.layers.Dense(latent, activation="linear", name="latent")(h)
    h = keras.layers.Dense(hidden, activation="relu", kernel_initializer=he,
                           kernel_regularizer=reg, name="dec_hidden")(z)
    h = keras.layers.Dropout(dropout, name="dec_dropout")(h)
    outputs = keras.layers.Dense(dim, activation="linear", name="reconstruction")(h)

    model = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, z, name="encoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    # Shuffle once with our seed so the validation split isn't just "the last 20% of the CSV".
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True)   # Unit 4: early stopping
    hist = model.fit(X[perm], X[perm], epochs=epochs, batch_size=batch_size,
                     validation_split=val_split, callbacks=[early], verbose=0, shuffle=True)

    # Stand-alone decoder (latent -> reconstruction) sharing the trained decoder layers.
    z_in = keras.Input(shape=(latent,))
    d = model.get_layer("dec_hidden")(z_in)
    d = model.get_layer("dec_dropout")(d)          # inactive at predict time
    decoder = keras.Model(z_in, model.get_layer("reconstruction")(d), name="decoder")

    val = [float(v) for v in hist.history["val_loss"]]
    recon = model.predict(X, verbose=0)
    return AutoencoderResult(
        latent=encoder.predict(X, verbose=0).astype(float),
        train_loss=[float(v) for v in hist.history["loss"]],
        val_loss=val,
        best_epoch=int(np.argmin(val)) + 1,
        epochs_run=len(val),
        reconstruction_mse=float(np.mean((recon - X) ** 2)),
        config=dict(hidden=hidden, latent=latent, dropout=dropout, l2=l2,
                    learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                    val_split=val_split, patience=patience, seed=seed),
        decode=lambda Z: decoder.predict(np.asarray(Z, dtype=float), verbose=0).astype(float),
    )
