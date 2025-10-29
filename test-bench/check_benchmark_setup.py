#!/usr/bin/env python3
"""
Quick verification script to check if benchmark dependencies and servers are ready
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False

def check_dependencies():
    """Check required Python packages"""
    packages = ['requests', 'psutil', 'openai', 'pandas', 'matplotlib']
    all_ok = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (install with: pip3 install {package})")
            all_ok = False
    
    return all_ok

def check_servers():
    """Check if NanoLLM and Ollama servers are accessible"""
    import requests
    
    servers = {
        'NanoLLM': 'http://localhost:8080/health',
        'Ollama': 'http://localhost:11434/api/tags'
    }
    
    results = {}
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name} server (running)")
                results[name] = True
            else:
                print(f"✗ {name} server (responded but error: {response.status_code})")
                results[name] = False
        except requests.exceptions.ConnectionError:
            print(f"✗ {name} server (not running)")
            results[name] = False
        except Exception as e:
            print(f"✗ {name} server (error: {e})")
            results[name] = False
    
    return results

def check_tegrastats():
    """Check if tegrastats is available"""
    try:
        result = subprocess.run(['which', 'tegrastats'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ tegrastats (available)")
            return True
        else:
            print("⚠ tegrastats (not available, will use psutil fallback)")
            return False
    except:
        print("⚠ tegrastats (not available, will use psutil fallback)")
        return False

def main():
    print("="*60)
    print("LLM Benchmark - System Check")
    print("="*60)
    
    print("\n1. Checking Python Version:")
    py_ok = check_python_version()
    
    print("\n2. Checking Python Dependencies:")
    deps_ok = check_dependencies()
    
    print("\n3. Checking Servers:")
    servers = check_servers()
    servers_ok = any(servers.values())
    
    print("\n4. Checking System Tools:")
    tegra_ok = check_tegrastats()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if py_ok and deps_ok and servers_ok:
        print("✓ System is ready for benchmarking!")
        print(f"\nAvailable backends: {', '.join([k for k, v in servers.items() if v])}")
        print("\nRun: python3 benchmark_llm.py")
    else:
        print("✗ System is NOT ready")
        if not py_ok:
            print("  - Upgrade Python to 3.10+")
        if not deps_ok:
            print("  - Install missing dependencies: pip3 install --user openai pandas matplotlib requests psutil")
        if not servers_ok:
            print("  - Start NanoLLM or Ollama server")
            print("    NanoLLM: jetson-containers run dustynv/nanollm:latest python3 -m nanollm.server")
            print("    Ollama:  ollama serve")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
