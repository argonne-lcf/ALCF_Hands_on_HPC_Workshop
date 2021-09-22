import ray
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]

def load_data():
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)

    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)

    return train_dataframe, val_dataframe


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = tf.keras.layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = (
        tf.keras.layers.StringLookup if is_string else tf.keras.layers.IntegerLookup
    )
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def run(config: dict):
    train_dataframe, val_dataframe = load_data()

    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)

    train_ds = train_ds.batch(config["batch_size"])
    val_ds = val_ds.batch(config["batch_size"])

    # Categorical features encoded as integers
    sex = tf.keras.Input(shape=(1,), name="sex", dtype="int64")
    cp = tf.keras.Input(shape=(1,), name="cp", dtype="int64")
    fbs = tf.keras.Input(shape=(1,), name="fbs", dtype="int64")
    restecg = tf.keras.Input(shape=(1,), name="restecg", dtype="int64")
    exang = tf.keras.Input(shape=(1,), name="exang", dtype="int64")
    ca = tf.keras.Input(shape=(1,), name="ca", dtype="int64")

    # Categorical feature encoded as string
    thal = tf.keras.Input(shape=(1,), name="thal", dtype="string")

    # Numerical features
    age = tf.keras.Input(shape=(1,), name="age")
    trestbps = tf.keras.Input(shape=(1,), name="trestbps")
    chol = tf.keras.Input(shape=(1,), name="chol")
    thalach = tf.keras.Input(shape=(1,), name="thalach")
    oldpeak = tf.keras.Input(shape=(1,), name="oldpeak")
    slope = tf.keras.Input(shape=(1,), name="slope")

    all_inputs = [
        sex,
        cp,
        fbs,
        restecg,
        exang,
        ca,
        thal,
        age,
        trestbps,
        chol,
        thalach,
        oldpeak,
        slope,
    ]

    # Integer categorical features
    sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
    cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
    fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
    restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
    exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
    ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)

    # String categorical features
    thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", train_ds)
    trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
    chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
    thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
    oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
    slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

    all_features = tf.keras.layers.concatenate(
        [
            sex_encoded,
            cp_encoded,
            fbs_encoded,
            restecg_encoded,
            exang_encoded,
            slope_encoded,
            ca_encoded,
            thal_encoded,
            age_encoded,
            trestbps_encoded,
            chol_encoded,
            thalach_encoded,
            oldpeak_encoded,
        ]
    )
    x = tf.keras.layers.Dense(config["units"], activation=config["activation"])(
        all_features
    )
    x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(all_inputs, output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer, "binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_ds, epochs=config["num_epochs"], validation_data=val_ds, verbose=0
    )

    return history.history["val_accuracy"][-1]


ACTIVATIONS = [
    "elu",
    "gelu",
    "hard_sigmoid",
    "linear",
    "relu",
    "selu",
    "sigmoid",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]

if __name__ == "__main__":

    n_gpus = len(get_available_gpus())
    is_gpu_available = n_gpus > 0

    if is_gpu_available:
        print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
    else:
        print("No GPU available")

    default_config = {
        "units": 32,
        "activation": "relu",
        "dropout_rate": 0.5,
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }

    # Avoid errors when multiple processes try to access the same GPU in parallel
    if is_gpu_available:
        run_default = ray.remote(num_cpus=1, num_gpus=1)(run)
    else:
        run_default = run
    objective_default = run_default(default_config)
    print("Accuracy Default Configuration: ", objective_default)

    from deephyper.problem import HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((8, 128), "units")
    problem.add_hyperparameter(ACTIVATIONS, "activation")
    problem.add_hyperparameter((0.0, 0.6), "dropout_rate")
    problem.add_hyperparameter((10, 100), "num_epochs")
    problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size")
    problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate")

    from deephyper.evaluator.evaluate import Evaluator

    evaluator = Evaluator.create(
        run, method="ray", method_kwargs={"num_cpus": 1, "num_cpus_per_task": 1}
    )

    from deephyper.search.hps import AMBS

    search = AMBS(problem, evaluator)

    results = search.search(max_evals=10)

    print(results)
