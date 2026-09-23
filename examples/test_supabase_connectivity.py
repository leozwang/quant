#!/usr/bin/env python3
"""Diagnostic tool to pinpoint Supabase connection issues on remote machines.

Run on the remote VM:
    python3 examples/test_supabase_connectivity.py
"""

import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

# Load dotenv if present
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from run_and_upload_recommendations import load_dotenv, SupabaseSync
    load_dotenv()
except Exception:
    pass

def step(title):
    print(f"\n{'=' * 60}\n▶ {title}\n{'=' * 60}")

def run_diagnostics():
    step("1. Environment & Proxy Variables")
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    print(f"  • SUPABASE_URL : {url or '[NOT SET]'}")
    print(f"  • SUPABASE_KEY : {key[:15]}...{key[-6:] if len(key) > 20 else '[NOT SET]'}")
    
    proxies = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    if proxies:
        print("  ⚠️  Active Proxy Variables detected (may intercept HTTPS):")
        for k, v in proxies.items():
            print(f"      {k} = {v}")
    else:
        print("  ✓ No proxy variables detected.")

    if not url:
        print("  ❌ SUPABASE_URL is not set. Stopping.")
        return

    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc or parsed.path
    port = parsed.port or 443

    step(f"2. DNS Resolution for {host}")
    try:
        addrinfo = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        ips = list({res[4][0] for res in addrinfo})
        print(f"  ✓ Resolved {host} to {len(ips)} IP(s): {', '.join(ips)}")
    except Exception as e:
        print(f"  ❌ DNS Resolution failed: {e}")
        return

    step(f"3. Raw TCP Connection to {host}:{port}")
    target_ip = ips[0]
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((target_ip, port))
        print(f"  ✓ TCP connection established with {target_ip}:{port}")
    except Exception as e:
        print(f"  ❌ TCP connection failed to {target_ip}:{port}: {e}")
        sock.close()
        return

    step("4. TLS / SSL Handshake")
    try:
        ctx = ssl.create_default_context()
        tls_sock = ctx.wrap_socket(sock, server_hostname=host)
        tls_sock.settimeout(5)
        cipher = tls_sock.cipher()
        version = tls_sock.version()
        print(f"  ✓ TLS Handshake successful! Protocol: {version}, Cipher: {cipher[0]}")
        tls_sock.close()
    except Exception as e:
        print(f"  ❌ TLS Handshake failed: {e}")
        return

    step("5. HTTP Request with default urllib (Python-urllib User-Agent)")
    req_url = f"{url}/rest/v1/qlib_daily_top10_recommend?limit=1"
    headers_base = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    try:
        req = urllib.request.Request(req_url, headers=headers_base)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            print(f"  ✓ Default urllib succeeded! HTTP {resp.status}, Body length: {len(data)} bytes")
    except urllib.error.HTTPError as e:
        print(f"  ⚠️  HTTP Error {e.code}: {e.read().decode('utf-8', errors='replace')}")
    except Exception as e:
        print(f"  ❌ Default urllib FAILED: {type(e).__name__}: {e}")
        print("     (This matches 'Remote end closed connection without response' if Cloudflare dropped it)")

    step("6. HTTP Request with Browser User-Agent + Connection: close")
    headers_custom = dict(headers_base)
    headers_custom["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers_custom["Connection"] = "close"
    headers_custom["Accept"] = "application/json"
    try:
        req2 = urllib.request.Request(req_url, headers=headers_custom)
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            data2 = resp2.read()
            print(f"  ✓ Custom User-Agent succeeded! HTTP {resp2.status}, Body length: {len(data2)} bytes")
            print(f"     Data preview: {data2[:100].decode('utf-8', errors='replace')}")
    except urllib.error.HTTPError as e:
        print(f"  ⚠️  HTTP Error {e.code}: {e.read().decode('utf-8', errors='replace')}")
    except Exception as e:
        print(f"  ❌ Custom User-Agent FAILED: {type(e).__name__}: {e}")

    step("7. Summary & Diagnosis")
    print("If Step 5 failed but Step 6 succeeded:")
    print("  -> Cloudflare is blocking Python's default User-Agent ('Python-urllib') from your VM IP.")
    print("If Step 3 or 4 failed:")
    print("  -> Firewall / MTU / Network egress issue on the VM.")
    print("If Step 1 showed proxy variables:")
    print("  -> An active proxy is intercepting/resetting connections.")

if __name__ == "__main__":
    run_diagnostics()
