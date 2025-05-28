#include <stdint.h>
#include <stdio.h>   // for printf

#define I2C_BASE   0x80006000
#define REG_CTRL   (*(volatile uint32_t *)(I2C_BASE + 0x00))
#define REG_TX     (*(volatile uint32_t *)(I2C_BASE + 0x04))
#define REG_CMD    (*(volatile uint32_t *)(I2C_BASE + 0x08))
#define REG_RX     (*(volatile uint32_t *)(I2C_BASE + 0x0C))

void init_platform() {}  // dummy stub for compatibility

void i2c_start() { REG_CMD = 0x01; }
void i2c_write(uint8_t d) { REG_TX = d; REG_CMD = 0x02; }
void i2c_read()  { REG_CMD = 0x04; }
void i2c_stop()  { REG_CMD = 0x08; }
uint8_t i2c_rx() { return REG_RX & 0xFF; }

void read_touch() {
    uint8_t buf[7];

    i2c_start();
    i2c_write(0x38 << 1);       // Write mode
    i2c_write(0x02);            // Start at reg 0x02
    i2c_start();
    i2c_write((0x38 << 1) | 1); // Read mode

    for (int i = 0; i < 7; i++) {
        i2c_read();
        buf[i] = i2c_rx();
    }
    i2c_stop();

    uint16_t x = ((buf[1] & 0x0F) << 8) | buf[2];
    uint16_t y = ((buf[3] & 0x0F) << 8) | buf[4];
    uint8_t touches = buf[0] & 0x0F;

    printf("Touches: %d | X: %d | Y: %d\n", touches, x, y);
}

int main() {
    init_platform();
    while (1) {
        read_touch();
        for (volatile int d = 0; d < 500000; d++);
    }
    return 0;
}

