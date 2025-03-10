import os
import subprocess


def open_tun(tun_name):
    # macOS verwendet /dev/tunX für TUN-Interfaces
    try:
        tun = open(f"/dev/{tun_name}", "r+b")
        return tun
    except FileNotFoundError:
        print(f"Kein TUN-Interface /dev/{tun_name} gefunden.")
        return None


def main():
    # TUN-Interface öffnen
    tun_name = "ttys000"  # Ändern Sie dies, falls Sie ein anderes Interface verwenden möchten
    tun = open_tun(tun_name)

    if tun:
        # Pakete lesen
        while True:
            packet = os.read(tun.fileno(), 2048)
            print(f"Paket empfangen: {packet.hex()}")
    else:
        print("Bitte stellen Sie sicher, dass ein TUN-Interface vorhanden ist.")


if __name__ == "__main__":
    main()
