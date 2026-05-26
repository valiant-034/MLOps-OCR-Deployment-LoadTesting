import requests

# ==========================================
# CHANGE THESE
# ==========================================

BASE_URL = " http://51.20.77.49:8000"
IMAGE_PATH = "/mnt/h/Semester-6/MLOps/Final_Deployment&Load_testing/test.png"

# ==========================================
# HEALTH CHECK
# ==========================================

try:
    response = requests.get(f"{BASE_URL}/health")

    print("\n=== HEALTH ENDPOINT ===")
    print("Status Code:", response.status_code)
    print("Response:", response.text)

except Exception as e:
    print("Health endpoint failed:", str(e))

# ==========================================
# PREDICTION CHECK
# ==========================================

try:
    with open(IMAGE_PATH, "rb") as img:

        files = {
            "image": ("test.png", img, "image/jpg")
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            files=files
        )

    print("\n=== PREDICT ENDPOINT ===")
    print("Status Code:", response.status_code)
    print("Response:", response.text)

except Exception as e:
    print("Predict endpoint failed:", str(e))