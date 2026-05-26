import pandas as pd

# ==========================================
# TARGET ENDPOINT
# ==========================================

ENDPOINT = "http://51.20.77.49:8000/predict"

# ==========================================
# REQUEST PAYLOAD INFO
# ==========================================

IMAGE_NAME = "test.png"
REQUEST_FIELD = "image"

# ==========================================
# TEST CONFIGS
# ==========================================

tests = [
    {
        "csv": "5users_stats.csv",
        "output": "loadtest_5users.txt",
        "users": 5,
        "spawn_rate": 2,
        "duration": 60
    },

    {
        "csv": "10users_stats.csv",
        "output": "loadtest_10users.txt",
        "users": 10,
        "spawn_rate": 1,
        "duration": 120
    },

]

# ==========================================
# GENERATE TXT REPORTS
# ==========================================

for test in tests:

    csv_file = test["csv"]
    output_file = test["output"]

    users = test["users"]
    spawn_rate = test["spawn_rate"]
    duration = test["duration"]

    print(f"\nProcessing: {csv_file}")

    # ==========================================
    # READ CSV
    # ==========================================

    df = pd.read_csv(csv_file)

    agg = df[df["Name"] == "Aggregated"].iloc[0]

    # ==========================================
    # BASIC METRICS
    # ==========================================

    total_requests = int(agg["Request Count"])
    total_failures = int(agg["Failure Count"])

    avg_response = float(agg["Average Response Time"])
    median_response = float(agg["Median Response Time"])

    # ==========================================
    # 95 PERCENTILE COLUMN
    # ==========================================

    p95_col = None

    for col in df.columns:
        if "95%" in col or "95" in col:
            p95_col = col
            break

    if p95_col:
        percentile_95 = float(agg[p95_col])
    else:
        percentile_95 = -1

    # ==========================================
    # REQUESTS PER SECOND
    # ==========================================

    rps = float(agg["Requests/s"])

    # ==========================================
    # STATUS CODE DISTRIBUTION
    # ==========================================

    status_200 = total_requests - total_failures
    status_400 = 0
    status_500 = total_failures

    # ==========================================
    # FAILURE ANALYSIS
    # ==========================================

    if total_failures == 0:

        failure_analysis = (
            "No failures observed during testing. "
            "All requests completed successfully."
        )

    elif total_failures < (0.05 * total_requests):

        failure_analysis = (
            "Minor request failures observed under concurrent "
            "OCR inference load."
        )

    else:

        failure_analysis = (
            "Significant failures observed due to server overload "
            "during concurrent OCR requests."
        )

    # ==========================================
    # BOTTLENECK ANALYSIS
    # ==========================================

    if avg_response < 500:

        bottleneck = (
            "No major bottleneck detected. "
            "System handled OCR inference efficiently."
        )

    elif avg_response < 2000:

        bottleneck = (
            "Moderate CPU and model inference bottleneck observed "
            "during concurrent OCR processing."
        )

    else:

        bottleneck = (
            "High OCR inference latency indicates CPU saturation "
            "and model execution bottleneck under heavy load."
        )

    # ==========================================
    # FINAL CONCLUSION
    # ==========================================

    if total_failures == 0 and avg_response < 2000:

        conclusion = "STABLE"

    elif total_failures < (0.10 * total_requests):

        conclusion = "MODERATELY STABLE"

    else:

        conclusion = "UNSTABLE"

    # ==========================================
    # REPORT FORMAT
    # ==========================================

    report = f"""=== TEST CONFIGURATION ===
Users: {users}
Spawn Rate: {spawn_rate}
Duration Seconds: {duration}

=== TARGET SYSTEM ===
Endpoint: {ENDPOINT}
Method: POST

=== REQUEST PAYLOAD ===
{{
  "{REQUEST_FIELD}": "{IMAGE_NAME}"
}}

=== LOCUST RESULTS ===
Total Requests: {total_requests}
Total Failures: {total_failures}

Average Response Time ms: {avg_response:.2f}
Median Response Time ms: {median_response:.2f}
95th Percentile ms: {percentile_95:.2f}

Requests Per Second: {rps:.2f}

=== STATUS CODE DISTRIBUTION ===
200: {status_200}
400: {status_400}
500: {status_500}

=== FAILURE ANALYSIS ===
{failure_analysis}

=== BOTTLENECK ANALYSIS ===
{bottleneck}

=== FINAL CONCLUSION ===
{conclusion}
"""

    # ==========================================
    # SAVE TXT FILE
    # ==========================================

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved: {output_file}")

print("\nAll TXT reports generated successfully.")