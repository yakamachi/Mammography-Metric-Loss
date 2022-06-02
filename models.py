import keras.models
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import applications

def classifierModel(in_model, class_count):
    classifier = tf.keras.models.Sequential()
    classifier.add(in_model)
    classifier.add(layers.Dense(class_count, name="output_layer_softmax", activation="softmax"))

    trainable = False
    for layer in classifier.layers:
        if layer.name == "output_layer_softmax":
            trainable = True
        layer.trainable = trainable

    return classifier

def createClassOnly(shape, class_count):
    input = layers.Input(name="input", shape=shape)
    res = applications.DenseNet201(weights=None, input_shape=shape, include_top=False)(input)
    flatten = layers.Flatten(name="flatten")(res)
    dense1 = layers.Dense(256, activation="relu", name="dense1")(flatten)
    norm1 = layers.BatchNormalization(name="norm1")(dense1)
    dense2 = layers.Dense(256, activation="relu", name="dense2")(norm1)
    norm2 = layers.BatchNormalization(name="norm2")(dense2)
    dense3 = layers.Dense(128, activation="relu", name="dense3")(norm2)
    norm3 = layers.BatchNormalization(name="norm3")(dense3)
    dense4 = layers.Dense(128, activation=None, name="dense4")(norm3)
    out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(dense4)
    dense5 = layers.Dense(64, activation="relu", name="class1")(out1)
    # dense5 = layers.Dense(32, activation="relu", name="class1")(out1)
    norm5 = layers.BatchNormalization(name="class2")(dense5)
    out2 = layers.Dense(class_count, name="output2", activation="softmax")(norm5)

    return Model(input, out2)

def createEmbModelForKNN(shape, class_count):
    input = layers.Input(name="input", shape=shape)
    res = applications.DenseNet201(weights=None, input_shape=shape, include_top=False)(input)
    flatten = layers.Flatten(name="flatten")(res)
    dense1 = layers.Dense(256, activation="relu", name="dense1")(flatten)
    norm1 = layers.BatchNormalization(name="norm1")(dense1)
    dense2 = layers.Dense(256, activation="relu", name="dense2")(norm1)
    norm2 = layers.BatchNormalization(name="norm2")(dense2)
    dense3 = layers.Dense(128, activation="relu", name="dense3")(norm2)
    norm3 = layers.BatchNormalization(name="norm3")(dense3)
    dense4 = layers.Dense(128, activation=None, name="dense4")(norm3)
    out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(dense4)

    return Model(input, out1)

def createFullModel(shape, class_count):

    # Embedding layers 1

    input = layers.Input(name="input", shape=shape)
    res = applications.DenseNet201(weights=None, input_shape=shape, include_top=False)(input)
    flatten = layers.Flatten(name="flatten")(res)
    dense1 = layers.Dense(256, activation="relu", name="dense1")(flatten)
    norm1 = layers.BatchNormalization(name="norm1")(dense1)
    dense2 = layers.Dense(256, activation="relu", name="dense2")(norm1)
    norm2 = layers.BatchNormalization(name="norm2")(dense2)
    dense3 = layers.Dense(128, activation="relu", name="dense3")(norm2)
    norm3 = layers.BatchNormalization(name="norm3")(dense3)
    dense4 = layers.Dense(128, activation=None, name="dense4")(norm3)
    out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(dense4)

    # Classification layers 2

    dense5 = layers.Dense(64, activation="relu", name="class1")(out1)
    norm5 = layers.BatchNormalization(name="class2")(dense5)
    out2 = layers.Dense(class_count, name="output2", activation="softmax")(norm5)

    return Model(input, [out1,out2])

def createFullModelHistory(shape, class_count):

    # Embedding layers 1

    input = layers.Input(name="input", shape=shape)
    # res = applications.resnet.ResNet50(weights=None, input_shape=shape, include_top=False)(input)
    res = applications.DenseNet201(weights=None, input_shape=shape, include_top=False)(input)
    # res = applications.MobileNetV3Small(weights=None, input_shape=shape, include_top=False)(input)
    # res = applications.InceptionV3(weights=None, input_shape=shape, include_top=False)(input)
    flatten = layers.Flatten(name="flatten")(res)
    dense1 = layers.Dense(256, activation="relu", name="dense1")(flatten)
    norm1 = layers.BatchNormalization(name="norm1")(dense1)
    dense2 = layers.Dense(256, activation="relu", name="dense2")(norm1)
    norm2 = layers.BatchNormalization(name="norm2")(dense2)
    dense3 = layers.Dense(128, activation="relu", name="dense3")(norm2)
    norm3 = layers.BatchNormalization(name="norm3")(dense3)
    dense4 = layers.Dense(128, activation=None, name="dense4")(norm3)
    #dense4 = layers.Dense(64, activation=None, name="dense4")(norm3)
    out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(dense4)

    # Embedding layers 2
    #
    # input = layers.Input(name="input", shape=shape)
    # # res = applications.resnet.ResNet50(weights=None, input_shape=shape, include_top=False)(input)
    # # res = applications.VGG16(weights=None, input_shape=shape, include_top=False)(input)
    # res = applications.xception.Xception(weights=None, input_shape=shape, include_top=False)(input)
    # # res = layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu")(input)
    # # res = layers.MaxPooling2D(pool_size=2)(res)
    # # res = layers.Dropout(0.3)(res)
    # # res = layers.Conv2D(filters=16, kernel_size=2, padding="same", activation="relu")(res)
    # # res = layers.MaxPooling2D(pool_size=2)(res)
    # # res = layers.Dropout(0.3)(res)
    # flatten = layers.Flatten(name="flatten")(res)
    # dense1 = layers.Dense(256, activation="relu", name="dense1")(flatten)
    # norm1 = layers.BatchNormalization(name="norm1")(dense1)
    # dense2 = layers.Dense(128, activation="relu", name="dense2")(norm1)
    # norm2 = layers.BatchNormalization(name="norm2")(dense2)
    # dense3 = layers.Dense(64, activation=None, name="dense3")(norm2)
    # #flatten = layers.Dense(256, activation=None)(dense3)
    # out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(dense3)

    # Embedding layers 3

    # input = layers.Input(name="input", shape=shape)
    # res = applications.resnet.ResNet50(weights=None, input_shape=shape, include_top=False)(input)
    # flatten = layers.Flatten(name="flatten")(res)
    # dense3 = layers.Dense(128, activation=None, name="dense3")(flatten)
    # out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(dense3)

    # Embedding layer 4

    # input = layers.Input(name="input", shape=shape)
    # res = applications.resnet.ResNet50(weights=None, input_shape=shape, include_top=False)(input)
    # res = layers.Flatten(name="flatten")(res)
    # res = layers.Dense(2000, activation="relu", name="multiperceptron")(res)
    # res = layers.BatchNormalization()(res)
    # res = layers.Dense(256, activation=None, name="dense")(res)
    # out1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(res)


    # Classification layers 1

    # out2 = layers.Dense(class_count, name="output2", activation="softmax")(out1)

    # Classification layers 2

    dense5 = layers.Dense(64, activation="relu", name="class1")(out1)
    #dense5 = layers.Dense(32, activation="relu", name="class1")(out1)
    norm5 = layers.BatchNormalization(name="class2")(dense5)
    out2 = layers.Dense(class_count, name="output2", activation="softmax")(norm5)

    # Classification layers 3

    # dense5 = layers.Dense(32, activation='relu', name='class1')(out1)
    # norm5 = layers.BatchNormalization(name='class2')(dense5)
    # dense6 = layers.Dense(16, activation='relu', name='class3')(norm5)
    # norm6 = layers.BatchNormalization(name='class4')(dense6)
    # out2 = layers.Dense(class_count, name="output2", activation="softmax")(norm6)

    return Model(input, [out1,out2])

def createEmbeddingModel(shape, trainable, weights):

    model = keras.models.Sequential([
    applications.resnet.ResNet50(weights=weights, input_shape=shape, include_top=False),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(128, activation=None),
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
        ])

    t = trainable

    for layer in model.layers:
        if layer.name == "conv5_block1_out":
            t = True
        layer.trainable = t

    return model

def createNetwork(in_model,shape):
    anchor = layers.Input(name="anchor", shape=shape)
    positive = layers.Input(name="positive", shape=shape)
    negative = layers.Input(name="negative", shape=shape)
    #anchor_emb = in_model(anchor)
    anchor_emb = in_model(applications.resnet.preprocess_input(anchor))
    #pos_emb = in_model(resnet.pre)
    pos_emb = in_model(applications.resnet.preprocess_input(positive))
    #neg_emb = in_model(negative)
    neg_emb = in_model(applications.resnet.preprocess_input(negative))
    output = layers.Lambda(distance)([anchor_emb,pos_emb,neg_emb])

    return Model(inputs=[anchor, positive, negative], outputs=output)

def distance(embeddings):
    an_po_dist = tf.math.reduce_sum(tf.math.square(embeddings[0] - embeddings[1]), -1)
    an_ne_dist = tf.math.reduce_sum(tf.math.square(embeddings[0] - embeddings[2]), -1)

    return an_po_dist, an_ne_dist

class SiameseModel(Model):

    def __init__(self, embedding_model, shape, margin=0.5):
        super().__init__()
        self.siamese_network = createNetwork(in_model=embedding_model,shape=shape)
        self.margin = margin
        self.loss_track = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients,self.siamese_network.trainable_weights))

        self.loss_track.update_state(loss)

        return {"loss": self.loss_track.result()}

    def test_step(self, data):
        loss = self.compute_loss(data)

        self.loss_track.update_state(loss)

        return {"loss": self.loss_track.result()}

    def compute_loss(self, data):

        ap, an = self.siamese_network(data)

        loss = ap - an
        loss = tf.math.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_track]

