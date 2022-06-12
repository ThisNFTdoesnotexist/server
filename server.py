import bottle
from bottle import route, run, template, request, static_file, response

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from matplotlib import cm
import uuid
import requests


import warnings
warnings.filterwarnings('ignore')

import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Conv2DTranspose
from keras.models import load_model

from tensorflow.compat.v1.keras.layers import BatchNormalization

class GAN():
    def __init__(self):
        self.img_shape = (128, 128, 3)
        
        self.noise_size = 100

        optimizer = Adam(0.0002,0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)
        
        self.discriminator.trainable = False
        
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined.summary()
        
    def build_discriminator(self):
        
        model = Sequential()

        model.add(Conv2D(128, (3,3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
        
    def build_generator(self):
        epsilon = 0.00001 # Small float added to variance to avoid dividing by zero in the BatchNorm layers.
        noise_shape = (self.noise_size,)
        
        model = Sequential()
        
        model.add(Dense(8*8*512, activation='linear', input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 512)))
        
        model.add(Conv2DTranspose(512, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(256, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(256, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(128, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(3, kernel_size=[4,4], strides=[1,1], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))

        # Standard activation for the generator of a GAN
        model.add(Activation("tanh"))
        
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)
    
    def train(self, epochs, batch_size=128, metrics_update=50, save_images=100, save_model=2000):

        X_train = np.array(images)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)
        
        mean_d_loss=[0,0]
        mean_g_loss=0

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)
            
            """p = np.random.permutation(batch_size)
            disc_input = np.concatenate([imgs,gen_imgs])[p]
            disc_output = np.concatenate([np.ones((half_batch, 1)), np.zeros((half_batch, 1))])[p]

            # Training the discriminator
            # The loss of the discriminator is the mean of the losses while training on authentic and fake images
            d_loss = self.discriminator.train_on_batch(disc_input, disc_output)"""
            
            d_loss = 0.5 * np.add(self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1))),
                                  self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1))))

            # Training the generator more than the discriminator because
            # the discriminator is having a very high accuracy
            for _ in range(3):
                noise = np.random.normal(0, 1, (batch_size, self.noise_size))

                valid_y = np.array([1] * (batch_size))
                g_loss = self.combined.train_on_batch(noise, valid_y)
            
            mean_d_loss[0] += d_loss[0]
            mean_d_loss[1] += d_loss[1]
            mean_g_loss += g_loss
            
            # We print the losses and accuracy of the networks every 200 batches mainly to make sure the accuracy of the discriminator
            # is not stable at around 50% or 100% (which would mean the discriminator performs not well enough or too well)
            if epoch % metrics_update == 0:
                print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, mean_d_loss[0]/metrics_update, 100*mean_d_loss[1]/metrics_update, mean_g_loss/metrics_update))
                mean_d_loss=[0,0]
                mean_g_loss=0
            
            # Saving 25 images
            if epoch % save_images == 0:
                self.save_images(epoch)
            
            # We save the architecture of the model, the weights and the state of the optimizer
            # This way we can restart the training exactly where we stopped
            if epoch % save_model == 0:
                self.generator.save("generator_%d" % epoch)
                self.discriminator.save("discriminator_%d" % epoch)
                
    # Saving 25 generated images to have a representation of the spectrum of images created by the generator
    def save_images(self, epoch):
        noise = np.random.normal(0, 1, (1, self.noise_size))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale from [-1,1] into [0,1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(1,1, figsize = (5,5))
        axs.imshow(gen_imgs[0])
        axs.axis('off')
        # for i in range(2):
        #     for j in range(5):
        #         axs[i,j]
        #         axs[i,j]

        plt.show()
        
        fig.savefig("BoredApes/Images_%s.png" % epoch)
        plt.close()
        
    def load_models(self, discriminator, generator):
        self.discriminator = tf.keras.models.load_model(discriminator)
        self.generator = tf.keras.models.load_model(generator)
        

# the decorator
def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        # set CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

        if bottle.request.method != 'OPTIONS':
            # actual request; reply with the actual response
            return fn(*args, **kwargs)

    return _enable_cors

@route('/')
@enable_cors
def index():
    id = str(uuid.uuid4())
    gan.save_images(id)
    return static_file("Images_" + id + ".png", root="./BoredApes")

@route('/uploadToIPFS/<id>')
@enable_cors
def upload(id):
    res = upload_to_ipfs("./BoredApes/Images_" + id + ".png")
    return res

@route('/generateImage')
@enable_cors
def generate():
    id = str(uuid.uuid4())
    gan.save_images(id)
    
    return {"uuid": id} #{"image": "http://thischimpdoesnotexist.com/image/" + id, "tokenId": id}

@route('/image/<id>')
@enable_cors
def image(id):
    if (id != ""):
        return static_file("Images_" + id + ".png", root="./BoredApes")
    else:
        return "Error"

@route('/metadata/<id>')
@enable_cors
def image(id):
    if (id != ""):
        return {"image": "http://thischimpdoesnotexist.com/image/" + id, "tokenId": id}
    else:
        return "Error"

def upload_to_ipfs(filepath):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

    payload={'pinataOptions': '{"cidVersion": 1}',
    'pinataMetadata': '{"name": "Test", "keyvalues": {"company": "Pinata"}}'}
    files=[
    ('file',('image.jpg',open(filepath,'rb'),'application/octet-stream'))
    ]
    headers = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJlOGY5M2VmNi0yZWQzLTRmNWQtOTRiYi0yYmRkYjA5MzhmZDkiLCJlbWFpbCI6IjIwQGRpY2suY20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGluX3BvbGljeSI6eyJyZWdpb25zIjpbeyJpZCI6IkZSQTEiLCJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MX0seyJpZCI6Ik5ZQzEiLCJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MX1dLCJ2ZXJzaW9uIjoxfSwibWZhX2VuYWJsZWQiOmZhbHNlLCJzdGF0dXMiOiJBQ1RJVkUifSwiYXV0aGVudGljYXRpb25UeXBlIjoic2NvcGVkS2V5Iiwic2NvcGVkS2V5S2V5IjoiMmE0ZjZmNjQ2OGRhZGZlNTA5ZjEiLCJzY29wZWRLZXlTZWNyZXQiOiJjZDhiMDA2MzYxMDg5YTcxOGY0NTI4Nzk0MTdmZDUxNzgzYmUyOGUwMjg4ZTE3MzU5MmExYWJmNGJhNzc0MmUzIiwiaWF0IjoxNjU1MDQwODc3fQ.6uPBTA95-s9n1bXTWF5iPKubH-St7PURJ2UkCbBEa_c'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    return response.text


global gan
gan=GAN()
gan.load_models("discriminator_3200", "generator_3200")

def main():
    bottle.run(host='0.0.0.0', port=8080, server='paste')

main()