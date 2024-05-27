from locust import HttpUser, between, task
import redis

ch = redis.Redis(
        host=:


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    @task(1)
    def index(self):
        self.client.post("/", json={"file": "./dog.jpeg", "filename": "dog.jpeg"})

    @task(3)
    def predict(
