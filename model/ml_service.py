import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from threading import Thread
from my_thread_class import my_custom_thread

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
        host='172.18.0.2',
        port=settings.REDIS_PORT,
        )

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = resnet50.ResNet50(include_top=True, weights='imagenet')


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    print('in predict')
    class_name = None
    pred_probability = None
    image_name = './uploads/' + image_name 
    img = image.load_img(image_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x_batch = np.expand_dims(x, axis=0)
    x_batch = resnet50.preprocess_input(x_batch)

    predis = model.predict(x_batch)
    tuple_results = resnet50.decode_predictions(predis, top=1)
    if len(tuple_results) != 0:
        print(f'tupleeeeee {tuple_results}')
        tuple_results = tuple_results[0][0]
        class_name = tuple_results[1]
        pred_probability = tuple_results[2]
    else:
        print("there is no class in the image")

    return class_name, pred_probability

def predict_and_store(image_name, id_image):
    print('in predict and store')
    class_name, pred_probability = predict(image_name)
    if class_name is not None and pred_probability is not None:
        db.lpush(id_image, json.dumps({"prediction":str(class_name), "score":str(pred_probability)}))
    else:
        print("error in inference")


def get_job_from_redis():
    print('in get_job_from redis')
    resul = db.brpop(keys=[settings.REDIS_QUEUE], timeout=10)
    if resul is not None:
        print(resul)
        resul = json.loads(resul[1].decode('utf-8'))
        image_name = resul['image_name']
        id_image = resul['id']
        Thread(target=predict_and_store, args=(image_name, id_image)).start()
        get_job_from_redis()
    else:
        print("there is no job in redis")
        get_job_from_redis()

    time.sleep(settings.SERVER_SLEEP)

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    print('classsify_process')
    try:
        thread_classify_process = Thread(target=get_job_from_redis)
        thread_classify_process.start()
    except Exception as e:
        print("a exception has ocurred!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(e)
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO
        #raise NotImplementedError

        # Sleep for a bit
        #time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
