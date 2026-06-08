"""
Estimates basic system capacity requirements.
Usage: python capacity_estimator.py <requests_per_second> <avg_payload_kb>
"""
import sys


def estimate(rps, payload_kb):
    daily_requests = rps * 86400
    daily_bandwidth_mb = (daily_requests * payload_kb) / 1024
    yearly_storage_tb = (daily_bandwidth_mb * 365) / (1024 * 1024)

    print("--- Capacity Estimate ---")
    print(f"Daily Requests:           {daily_requests:,}")
    print(f"Daily Bandwidth:          {daily_bandwidth_mb:,.2f} MB")
    print(f"Yearly Storage (if saved): {yearly_storage_tb:,.2f} TB")
    print(f"Recommended instances (stateless, 2k RPS each): {max(1, rps // 2000)}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        estimate(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Usage: python capacity_estimator.py <rps> <payload_kb>")
        print("Example: python capacity_estimator.py 500 50")
