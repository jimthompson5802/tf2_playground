#
#  Prototype for Custom Model with custom training loop
#

from collections import OrderedDict, namedtuple
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

#
# make repeatable
#
np.random.seed(113)
tf.random.set_seed(113)


# emulates Ludwig's Input Sequence Feature w/ LSTM encoder
class CustomInputSequence(layers.Layer):
    def __init__(self, num_words=16, embed_size=32, lstm_units=32,
                 name='sequence_input'):
        super(CustomInputSequence, self).__init__(name=name)

        self.num_words = num_words

        self.input_layer = layers.Input(
            shape=[None, ], dtype='int32',
            name='input_layer_{}'.format(name)
        )
        self.embedding = layers.Embedding(num_words, embed_size)
        self.lstm = layers.LSTM(lstm_units)

    # emulates feature encoder
    def encoder(self, inputs, training=None, mask=None):
        features = self.embedding(inputs, training=training)
        features = self.lstm(features, training=training)
        return features

    def call(self, inputs, training=None, mask=None):

        encoder_output = self.encoder(inputs, training=training, mask=None)

        return {
            'encoder_output': encoder_output
        }


# Emulates Ludwig's Input Category Feature w/ pass through encoder
class CustomInputCategory(layers.Layer):
    def __init__(self, num_tags=16,
                 name='category_input'):
        super(CustomInputCategory, self).__init__(name=name)

        self.num_tags = num_tags

        self.input_layer = layers.Input(
            shape=[num_tags, ], dtype='float32',
            name='input_layer_{}'.format(name)
        )

    # emulates encoder
    def encoder(self, inputs, training=None, mask=None):
        return tf.convert_to_tensor(inputs, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):

        encoder_output = self.encoder(inputs, training=training, mask=None)

        return {
            'encoder_output': encoder_output
        }


# emulates Ludwig's Numeric Input Feature
class CustomInputNumeric(tf.keras.layers.Layer):
    def __init__(self, name='if_name'):
        super(CustomInputNumeric, self).__init__()

        self.input_layer = layers.Input(
            shape=[None, 1], dtype='float32',
            name='input_layer_{}'.format(name)
        )

    # emulates input feature encoder
    def encoder(self, inputs, training=None):
        return tf.convert_to_tensor(inputs, dtype=tf.float32)

    def call(self, inputs, training=None):
        encoder_output = self.encoder(inputs, training=training)
        return {
            'encoder_output': encoder_output
        }


# emulates Ludwig's Numeric Output Feature
class CustomOutputNumeric(tf.keras.layers.Layer):
    def __init__(self, fc_layers, fc_size, name='of_name'):
        super(CustomOutputNumeric, self).__init__()

        self.logits = tf.zeros([1])
        self.fc_layers = fc_layers
        self.layers = []
        for _ in range(fc_layers):
            self.layers.append(
                tf.keras.layers.Dense(
                    fc_size, activation=tf.keras.activations.relu
                )
            )
        self.regressor_decoder = tf.keras.layers.Dense(
            1,
            activation=None
        )

    # emulates the output feature decoder
    def decoder(self, input_tensor, training=None, **kwargs):
        hidden = input_tensor
        for n in range(self.fc_layers):
            hidden = self.layers[n](hidden, training=training)
        logits = self.regressor_decoder(hidden, training=training)
        return logits

    def call(self, inputs, training=None, **kwargs):
        hidden = inputs['combiner_output']
        self.logits = self.decoder(hidden, training=training)
        return self.logits


# Emulates Ludwig's Binary Output Feature
class CustomOutputBinary(keras.layers.Layer):
    def __init__(self, name='output_binary'):
        super(CustomOutputBinary, self).__init__(name=name)

        self.binary = layers.Dense(1)
        self.logits = tf.zeros([1])
        self.loss_function = keras.losses.BinaryCrossentropy(from_logits=True)

    # emulates decoder
    def decoder(self, inputs, training=None, mask=None):
        return self.binary(inputs)

    def call(self, inputs, training=None, mask=None):
        hidden = inputs['combiner_output']
        self.logits = self.decoder(hidden, training=training)
        return self.logits


# Emulates Ludwig's category Output Feature
class CustomOutputCategory(keras.layers.Layer):
    def __init__(self, num_classes, name='output_binary'):
        super(CustomOutputCategory, self).__init__(name=name)

        self.category = layers.Dense(num_classes)
        self.logits = tf.zeros([1])
        self.loss_function = keras.losses.CategoricalCrossentropy(from_logits=True)

    # emulates decoder
    def decoder(self, inputs, training=None, mask=None):
        return self.category(inputs)

    def call(self, inputs, training=None, mask=None):
        hidden = inputs['combiner_output']
        self.logits = self.decoder(hidden, training=training)
        return self.logits

# Emulates Ludwig's Combiner
class CustomCombiner(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomCombiner, self).__init__()

    def call(self, inputs, training=None):
        encoder_outputs = []
        for k in inputs:
            encoder_outputs.append(inputs[k]['encoder_output'])

        if len(inputs) > 1:
            hidden = tf.keras.layers.concatenate(encoder_outputs, 1)
        else:
            hidden = encoder_outputs[0]

        return {
            'combiner_output': hidden
        }


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, feature_name):
        super(CustomLoss, self).__init__()

        self.feature_name = feature_name
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        y = y_true[self.feature_name]
        y_hat = y_pred[self.feature_name]

        loss = self.loss_function(y, y_hat)
        return loss


# emulates Ludwig's ECD class
class CustomModel(keras.models.Model):

    def __init__(self):

        super(CustomModel, self).__init__()

        self.input_features = OrderedDict([
            ('x1', CustomInputNumeric(name='x1')),
            ('x2', CustomInputNumeric(name='x2'))
        ])

        self.combiner_layer = CustomCombiner()

        self.output_features = OrderedDict([
            ('y', CustomOutputNumeric(5, 64, name='y'))
        ])

        self.input_tensors = []
        for inp in self.input_features:
            self.input_tensors.append(self.input_features[inp].input_layer)

        self.output_logits = []
        for otp in self.output_features:
            self.output_logits.append(self.output_features[otp].logits)

        self.inputs = self.input_tensors
        self.outputs = self.output_logits

        self.train_loss = CustomLoss('y')
        self.metric_fn = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs, training=None, mask=None):

        encoder_outputs = {}

        for if_name in self.input_features:
            encoder_output = self.input_features[if_name](inputs[if_name],
                                                          training=training)
            encoder_outputs[if_name] = encoder_output

        combiner_outputs = self.combiner_layer(encoder_outputs,
                                               training=training)
        results = {}
        for of_name in self.output_features:
            logits = self.output_features[of_name](combiner_outputs,
                                                   training=training)
            results[of_name] = logits

        return results

    def train_step(self, inputs, training=None, mask=None):
        x, y = inputs

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.train_loss(y, y_pred)

        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        self.metric_fn.update_state(loss)

    def predict_step(self, inputs):
        preds = self(inputs[0], training=False)
        return preds


tf.config.experimental_run_functions_eagerly(True)
num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

# Instantiate an end-to-end model predicting both priority and department
model = CustomModel()

# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss={
#         "y": keras.losses.MeanSquaredError()
#     }
# )

#
# Helper functions
#
GeneratedData = namedtuple('GeneratedData',
                           'train test')
def generate_data():
    # function generates simple training data that guarantee convergence
    NUMBER_OBSERVATIONS = 500

    # generate data
    e1 = np.random.normal(size=NUMBER_OBSERVATIONS).reshape(-1, 1)
    x1 = np.array(range(NUMBER_OBSERVATIONS)).reshape(-1, 1) + e1

    e2 = np.random.normal(size=NUMBER_OBSERVATIONS).reshape(-1, 1)
    x2 = np.random.choice(range(NUMBER_OBSERVATIONS), size=NUMBER_OBSERVATIONS, replace=False).reshape(-1, 1) + e2

    y = 2*x1 + 3*x2 + 1 + np.random.normal(size=NUMBER_OBSERVATIONS).reshape(-1, 1)

    # create train/test data sets
    train_x1, test_x1, train_x2, test_x2, train_y, test_y  = train_test_split(
        x1,
        x2,
        y,
        train_size=0.7
    )

    return GeneratedData(
        {
            # training data set
            'x1': train_x1.astype(np.float32),
            'x2': train_x2.astype(np.float32),
            'y': train_y.astype(np.float32)
        },
        {
            # test data set
            'x1': test_x1.astype(np.float32),
            'x2': test_x2.astype(np.float32),
            'y': test_y.astype(np.float32)
        }
    )

# quick and dirty generator to return random batches from training data
def batcher(total_size, batch_size):
    i = 0
    idx = np.random.permutation(range(total_size))
    for i in range(0, total_size, batch_size):
        yield list(idx[i:(i + batch_size)])

#
# generate training and test data
#
generated_data = generate_data()

# model.fit(
#     {"x1": generated_data.train['x1'], 'x2': generated_data.train['x2']},
#     {"y": generated_data.train['y']},
#     epochs=10,
#     batch_size=16
# )

# get size from first feature in training data, assume others same size
training_size = generated_data.train['x1'].size

for epoch in range(10):
    model.metric_fn.reset_states()
    batches = batcher(training_size, 16)
    for batch in batches:
        inp_feat = {
            'x1': generated_data.train['x1'][batch],
            'x2': generated_data.train['x2'][batch]
        }
        out_feat = {
            'y': generated_data.train['y'][batch]
        }
        inputs = (inp_feat, out_feat)
        model.train_step(inputs, training=True)

    print("completed epoch {:d}, loss: {:.5f}"
            .format(epoch+1, model.metric_fn.result().numpy())
          )


    test_size = generated_data.test['x1'].size
    batches = batcher(test_size, 32)
    preds = []
    for batch in batches:
        inp_feat = {
            'x1': generated_data.test['x1'][batch],
            'x2': generated_data.test['x2'][batch]
        }
        preds.append(model.predict_step((inp_feat,))['y'])

    predictions = tf.concat(preds, axis=0)


preds1 = pd.DataFrame(np.concatenate([generated_data.test['y'], predictions.numpy()], axis=1))
preds1.columns = ['y_true', 'y_pred1']
print(preds1.head(10))

weights1 = []
for v in model.variables:
    weights1.append(v)

#
# Save model
#
shutil.rmtree('./saved_model', ignore_errors=True)
model.save('./saved_model')
del(model)

#
# load up saved model
#
reloaded_model = tf.saved_model.load('./saved_model')
print("object type for reloaded custom model:", type(reloaded_model))

print("predicting with reloaded model")
preds2 = reloaded_model(
    {'x1': generated_data.test['x1'], 'x2': generated_data.test['x2']},  #inputs
    False,  #training
    None    #mask
)
preds2 = pd.DataFrame(np.concatenate([generated_data.test['y'], preds2['y']], axis=1))
preds2.columns = ['y_true', 'y_pred1']
print(preds2.head(10))

weights2 = []
for v in reloaded_model.variables:
    weights2.append(v)

print("before/after predictions matched:", np.all(np.isclose(preds1, preds2)))

print(
    "before/after variable matched:",
      np.all(
          [np.all(np.isclose(w1.numpy(), w2.numpy())) for w1, w2 in zip(weights1, weights2)]
      )
)

