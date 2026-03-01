
import sys
import os
sys.path.append(os.getcwd())
try:
    from questdb.ingress import Sender
except ImportError:
    print("Could not import Sender.")
    sys.exit(1)

def test_sender():
    host = '127.0.0.1'
    port = 9009
    
    print(f"Testing Sender({host}, {port})...")
    try:
        with Sender(host, port) as sender:
            pass
        print("Success: Sender(host, port)")
        return
    except Exception as e:
        print(f"Failed: Sender(host, port) - {e}")

    print(f"Testing Sender('tcp', {host}, {port})...")
    try:
        # Assuming protocol as first arg?
        # But 'tcp' is string. 
        # Actually, maybe it takes a SINGLE string argument (conf_str)?
        conf = f"http::addr={host}:{port};" # Start with http
        # But ILP uses TCP.
        conf_tcp = f"tcp::addr={host}:{port};"
        
        with Sender.from_conf(conf_tcp) as sender:
             pass
        print(f"Success: Sender.from_conf('{conf_tcp}')")
    except AttributeError:
        print("Sender.from_conf does not exist.")
    except Exception as e:
        print(f"Failed: Sender.from_conf('{conf_tcp}') - {e}")

    try:
        # Maybe it takes `auth` as 3rd arg?
        with Sender(host, port, None) as sender:
            pass
        print("Success: Sender(host, port, None)")
    except Exception as e:
        print(f"Failed: Sender(host, port, None) - {e}")

if __name__ == "__main__":
    test_sender()
