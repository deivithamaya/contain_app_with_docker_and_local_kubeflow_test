from locust import HttpUser, between, task
import redis
from werkzeug.datastructures import FileStorage

ch = redis.Redis(
        host="172.2.0.2",
        port=6379
        )

class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    #@task(1)
    #def index(self):
    #    self.client.post("/", json={"file": "./dog.jpeg", "filename": "dogman.jpeg"})

    @task(3)
    def predict(self):
        print("predict")
        file = None
        with open('./doglocust.jpeg', 'rb') as fp:
            self.client.post("/predict", files={'file':fp, 'filename':"doglocust.jpeg"})

