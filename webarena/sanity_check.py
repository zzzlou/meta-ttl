#!/usr/bin/env python3
import requests
import time

# Configure the server IP and test URL for each service.
HOST = "localhost"
SERVICES = {
    "Shopping": f"http://{HOST}:8082/",
    "Shopping Admin": f"http://{HOST}:8083/admin",
    "Reddit": f"http://{HOST}:8080",
    "GitLab": f"http://{HOST}:8084/users/sign_in",
    "Wikipedia": f"http://{HOST}:8081/",
    "Map": f"http://{HOST}:443",
    "Reset Server": f"http://{HOST}:8085"
}

def check_services():
    print(f"{'='*50}")
    print(f"WebArena connectivity check (Target: {HOST})")
    print(f"{'='*50}\n")
    
    all_passed = True

    for name, url in SERVICES.items():
        try:
            # Use `requests.head` to mimic `curl -I`.
            start_time = time.time()
            response = requests.head(url, timeout=5, allow_redirects=True)
            elapsed = time.time() - start_time
            
            status = response.status_code
            
            # Any valid HTTP response code, including 404/405, still proves connectivity.
            if status in [200, 301, 302, 404, 405]:
                print(f"✅ [PASS] {name:.<18} HTTP {status} ({elapsed:.2f}s)")
            elif status == 502:
                print(f"⚠️  [WARN] {name:.<18} HTTP 502 (service is still starting, please wait a few minutes)")
                all_passed = False
            else:
                print(f"❓ [UNKNOWN] {name:.<15} HTTP {status} ({elapsed:.2f}s)")
                all_passed = False
                
        except requests.exceptions.Timeout:
            print(f"❌ [FAIL] {name:.<18} Timeout (likely blocked by a firewall, or the IP may be incorrect)")
            all_passed = False
        except requests.exceptions.ConnectionError:
            print(f"❌ [FAIL] {name:.<18} Connection Refused (the service is probably not running on the server)")
            all_passed = False
        except Exception as e:
            print(f"❌ [FAIL] {name:.<18} Error: {str(e)}")
            all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 Perfect! All services are reachable and you're ready to run the agent.")
    else:
        print("🚨 Some services are still unreachable. Please troubleshoot using the errors above.")
    print(f"{'='*50}")

if __name__ == "__main__":
    check_services()
