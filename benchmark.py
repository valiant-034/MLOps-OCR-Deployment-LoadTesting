import subprocess

# ==========================================
# CHANGE THIS
# ==========================================

HOST = "http://51.20.77.49:8000"

# ==========================================
# TEST CONFIGS
# ==========================================

tests = [
    {
        "users": 5,
        "spawn_rate": 2,
        "duration": "60s",
        "name": "5users"
    },

    {
        "users": 10,
        "spawn_rate": 1,
        "duration": "120s",
        "name": "10users"
    },

]

# ==========================================
# RUN TESTS
# ==========================================

for test in tests:

    users = test["users"]
    spawn_rate = test["spawn_rate"]
    duration = test["duration"]
    name = test["name"]

    print("\n====================================")
    print(f"Running benchmark: {name}")
    print("====================================\n")

    command = [
        "locust",
        "--locustfile=locustfile.py",
        "--headless",
        f"--users={users}",
        f"--spawn-rate={spawn_rate}",
        f"--run-time={duration}",
        f"--host={HOST}",
        f"--csv={name}",
        f"--html={name}.html"
    ]

    subprocess.run(command)

print("\nAll benchmarks completed.")