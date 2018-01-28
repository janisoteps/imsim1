import numpy.random as rng
# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
import tensorflow as tf
sess = tf.Session()


def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)


# initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


# def loss(x1, x2, y):
#     # Euclidean distance between x1,x2
#     l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(x1, x2)),
#                                     reduction_indices=1))
#
#     # you can try margin parameters
#     margin = tf.constant(1.)
#
#     labels = tf.to_float(y)
#
#     match_loss = tf.square(l2diff, 'match_term')
#     mismatch_loss = tf.maximum(0., tf.sub(margin, tf.square(l2diff)), 'mismatch_term')
#
#     # if label is 1, only match_loss will count, otherwise mismatch_loss
#     loss = tf.add(tf.mul(labels, match_loss), tf.mul((1 - labels), mismatch_loss), 'loss_add')
#
#     loss_mean = tf.reduce_mean(loss)
#     return loss_mean

input_shape = (1, 4096)
left_input = Input(input_shape)
right_input = Input(input_shape)

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(512, activation='relu', input_shape=(1, 4096)))

# Add one hidden layer
model.add(Dense(512, activation='relu'))

# Add an output layer
model.add(Dense(128, activation='sigmoid'))

#encode each of the two inputs into a vector with the convnet
encoded_l = model(left_input)
encoded_r = model(right_input)

#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0]-x[1])
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)


optimizer = Adam(0.00006)
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()

# model.compile(loss=loss(x1_, x2_, y_),
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)