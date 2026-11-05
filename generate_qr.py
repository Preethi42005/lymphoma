#!/usr/bin/env python3
"""
Generate QR code for LymphoScan project demonstration
"""

import qrcode
from qrcode.constants import ERROR_CORRECT_H

# Project URL - direct to dashboard (network accessible)
PROJECT_URL = "http://10.200.64.185:8080/dashboard"

# Create QR code with high error correction
qr = qrcode.QRCode(
    version=1,
    error_correction=ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

qr.add_data(PROJECT_URL)
qr.make(fit=True)

# Create QR code image
img = qr.make_image(fill_color="black", back_color="white")

# Save the QR code
img.save("lymphoscan_qr.png")
print(f"QR code generated successfully!")
print(f"URL: {PROJECT_URL}")
print(f"Saved as: lymphoscan_qr.png")
