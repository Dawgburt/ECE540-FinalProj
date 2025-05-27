import tkinter as tk
import smbus2
import time
import threading

# FT6336U I2C address
I2C_ADDR = 0x38
bus = smbus2.SMBus(1)

def read_touch_data():
    try:
        # Read number of touch points
        touch_points = bus.read_byte_data(I2C_ADDR, 0x02) & 0x0F
        if touch_points == 0:
            return None

        # Read coordinates from register 0x03
        data = bus.read_i2c_block_data(I2C_ADDR, 0x03, 6)

        x = ((data[0] & 0x0F) << 8) | data[1]
        y = ((data[2] & 0x0F) << 8) | data[3]
        return (x, y)

    except Exception as e:
        return f"Error: {e}"

class TouchApp:
    def __init__(self, root):
        self.root = root
        root.title("FT6336U Touch Viewer")

        self.label = tk.Label(root, text="Waiting for touch...", font=("Arial", 24))
        self.label.pack(padx=20, pady=40)

        self.running = True
        self.update_thread = threading.Thread(target=self.update_touch_loop, daemon=True)
        self.update_thread.start()

    def update_touch_loop(self):
        while self.running:
            touch = read_touch_data()
            if isinstance(touch, tuple):
                self.label.config(text=f"Touch: x = {touch[0]}, y = {touch[1]}")
            elif isinstance(touch, str):
                self.label.config(text=touch)
            else:
                self.label.config(text="No touch")
            time.sleep(0.05)

    def stop(self):
        self.running = False

# Start the app
if __name__ == "__main__":
    root = tk.Tk()
    app = TouchApp(root)
    try:
        root.mainloop()
    finally:
        app.stop()
