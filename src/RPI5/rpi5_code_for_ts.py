import time
import smbus2
import numpy as np
from PIL import Image, ImageDraw
import gpiod
import subprocess

# FT6336U I2C touchscreen settings
bus = smbus2.SMBus(1)
addr = 0x38
timeout = 1.0  # seconds of no touch to stop recording
width, height = 320, 480

# GPIO Protocol mapping (BCM numbers)
LINE_MAP = {
    0: 26, 1: 16, 2: 6, 3: 5, 4: 24, 5: 23, 6: 22, 7: 12,  # Pixel data bits
    8: 21,  # data_valid (output)
    9: 20,  # data_ack (input from FPGA)
    10: 13  # end_of_image (output)
}

CHIP_NAME = "gpiochip4"  # Valid for GPIOs 128–159 (RPi5, confirmed)

# Touchscreen Reset Pin (GPIO17 = BCM 17 = physical pin 11)
RESET_GPIO = 17
RESET_CHIP = "gpiochip0"  # GPIO17 is on gpiochip0 on RPi5

def reset_touchscreen():
    chip = gpiod.Chip(RESET_CHIP)
    line = chip.get_line(RESET_GPIO)
    line.request(consumer="ctp_rst", type=gpiod.LINE_REQ_DIR_OUT)

    print("Toggling GPIO17 for CTP_RST...")
    line.set_value(0)
    time.sleep(0.05)
    line.set_value(1)
    time.sleep(0.05)

    line.release()
    chip.close()

# Perform reset first
reset_touchscreen()

# Open GPIO lines for image sending
chip = gpiod.Chip(CHIP_NAME)
output_lines = [LINE_MAP[i] for i in range(8)] + [LINE_MAP[8], LINE_MAP[10]]
lines_out = chip.get_lines(output_lines)
lines_out.request(consumer="send_image", type=gpiod.LINE_REQ_DIR_OUT)

ack_line = chip.get_line(LINE_MAP[9])
ack_line.request(consumer="ack_wait", type=gpiod.LINE_REQ_DIR_IN)

# Touchscreen reader
def read_touch():
    try:
        touches = bus.read_byte_data(addr, 0x02)
        if touches == 0 or touches > 2:
            return None
        data = bus.read_i2c_block_data(addr, 0x03, 4)
        x = ((data[0] & 0x0F) << 8) | data[1]
        y = ((data[2] & 0x0F) << 8) | data[3]
        return (x, y)
    except:
        return None

# Send one pixel to FPGA
def send_pixel(value, is_last=False):
    bits = [(value >> i) & 1 for i in range(8)]
    lines_out.set_values(bits + [1, int(is_last)])  # data_valid = 1
    while ack_line.get_value() == 0:
        time.sleep(0.0001)
    lines_out.set_values(bits + [0, int(is_last)])  # data_valid = 0
    while ack_line.get_value() == 1:
        time.sleep(0.0001)

# Record touch gesture
print("Waiting for gesture...")
points = []
recording = False
last_touch_time = None

while True:
    point = read_touch()
    now = time.time()
    if point and 0 <= point[0] < width and 0 <= point[1] < height:
        if not recording:
            print("Touch started")
            points = []
            recording = True
        points.append(point)
        last_touch_time = now
    elif recording and last_touch_time and (now - last_touch_time > timeout):
        print("Touch ended, processing...")
        break
    time.sleep(0.03)

if not points:
    print("No gesture captured.")
    lines_out.release()
    ack_line.release()
    exit()

# Normalize drawing to 28x28
def normalize_and_draw(points, size=28):
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1)
    dy = max(max_y - min_y, 1)
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    scaled = [
        (
            int((x - min_x) / dx * (size - 1)),
            int((y - min_y) / dy * (size - 1))
        ) for x, y in points
    ]
    for i in range(1, len(scaled)):
        draw.line([scaled[i - 1], scaled[i]], fill=255)
    return img

# Show and send image
image = normalize_and_draw(points)
image.save("gesture.png")
subprocess.run(["feh", "gesture.png"])
print("Saved gesture as gesture.png")

pixels = np.array(image).flatten()
for i, val in enumerate(pixels):
    send_pixel(int(val), is_last=(i == 783))

lines_out.release()
ack_line.release()
print("✅ Done sending 28x28 frame to FPGA")

