from locust import HttpUser, task, between

IMAGE_PATH = "test.png"

class ModelUser(HttpUser):

    # Base URL
    host = "http://51.20.77.49:8000"

    # Wait between requests
    wait_time = between(3, 5)

    @task
    def predict(self):

        with open(IMAGE_PATH, "rb") as img:

            files = {
                "image": (
                    "test.png",
                    img,
                    "image/png"
                )
            }

            self.client.post(
                "/predict",
                files=files,
                timeout=30
            )

    @task
    def health(self):

        self.client.get(
            "/health",
            timeout=10
        )