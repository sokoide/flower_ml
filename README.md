# flower categorization

## flickr.py

* Change ```flickr_api_key``` and ```secret_key``` (optionally ```num_photos``` and ```keywords```) and run it
* It'll download photos from flicker into ./images/$keyword for given keywords (sakufa, sunflower, rose by default)

```bash
python3 flicker.py
```


## flower.py

* It'll fit (training) and save the model as flower.model, weights as flower.weights

```bash
python3 flower.py fit --epochs $num-epochs

i.e.
python3 flower.py fit --epochs 10
10/10 [==============================] - 31s 3s/step - loss: 1.0925 - acc: 0.3682
Epoch 2/10
10/10 [==============================] - 30s 3s/step - loss: 1.0663 - acc: 0.4642
Epoch 3/10
10/10 [==============================] - 28s 3s/step - loss: 1.0337 - acc: 0.5463
Epoch 4/10
10/10 [==============================] - 28s 3s/step - loss: 0.9627 - acc: 0.6264
Epoch 5/10
10/10 [==============================] - 28s 3s/step - loss: 0.8254 - acc: 0.6701
Epoch 6/10
10/10 [==============================] - 28s 3s/step - loss: 0.6723 - acc: 0.7124
Epoch 7/10
10/10 [==============================] - 29s 3s/step - loss: 0.5961 - acc: 0.7577
Epoch 8/10
10/10 [==============================] - 28s 3s/step - loss: 0.4921 - acc: 0.8199
Epoch 9/10
10/10 [==============================] - 28s 3s/step - loss: 0.4517 - acc: 0.8348
Epoch 10/10
10/10 [==============================] - 28s 3s/step - loss: 0.3216 - acc: 0.8930

(It took ~5 mins on 2016 MacBook Pro 13 inch)
```

* predict using the model and weights

```bash
python3 flower.py predict --image-path $path-to-image

i.e.
python3 flower.py predict --image-path ./images/sakura/13024582963.jpg
Using TensorFlow backend.
making or loading a model...
loading the model...
model loaded
model created/loaded
...

result: [[0.0190797  0.97461957 0.00630072]]
predicted: sakura, 97.46195673942566%
```
