import time

log_file = "/var/log/snort/alert"

snort_alerts = [
    "[**] [1:1000001:1] ET SCAN Potential DDoS [**]",
    "[Priority: 2]",
    "01/15-14:23:01.123456 192.168.1.100:54321 -> 10.0.0.5:80",
    "TCP TTL:64 TOS:0x0 ID:12345 IpLen:20 DgmLen:1460 DF",
    "******S* Seq: 0x...  Ack: 0x...  Win: 0x...  TcpLen: 40",
    "",
]

print(f"Generating realistic DDoS traffic to {log_file}...")
with open(log_file, "a", encoding="utf-8") as f:
    for i in range(50):
        for line in snort_alerts:
            # vary the DgmLen to match realistic max packet sizes for DDoS
            if "IpLen:" in line:
                line = line.replace("DgmLen:1460", f"DgmLen:{1400+(i%10)}")
            f.write(line + "\n")
        f.flush()
        time.sleep(0.05) # 50ms delay means 50 packets take ~2.5 seconds
print(f"Appended {len(snort_alerts)*50} lines to {log_file}")
