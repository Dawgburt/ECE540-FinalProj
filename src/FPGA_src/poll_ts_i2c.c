// Code for talking to the FT6336U over I2c
// Current State (5/31/2025): response is 0x03, but that means the touchscreen is off, so its a response but not what we want? more research is needed

#include <stdint.h>
#include "config_uart.h"
#include "uart.h"   // or ee_printf if using that

#define I2C_BASE 0x80006000

#define I2C_PRER_LO (*(volatile uint8_t *)(I2C_BASE + 0x00))
#define I2C_PRER_HI (*(volatile uint8_t *)(I2C_BASE + 0x01))
#define I2C_CTR     (*(volatile uint8_t *)(I2C_BASE + 0x02))
#define I2C_TXR     (*(volatile uint8_t *)(I2C_BASE + 0x03))
#define I2C_RXR     (*(volatile uint8_t *)(I2C_BASE + 0x03))
#define I2C_CR      (*(volatile uint8_t *)(I2C_BASE + 0x04))
#define I2C_SR      (*(volatile uint8_t *)(I2C_BASE + 0x04))

#define FT6336U_ADDR       0x38
#define FT6336U_CHIPID_REG 0xA3

void delay(int n) { while (n--) asm volatile("nop"); }

void i2c_wait() {
    while (I2C_SR & 0x02);  // Wait for TIP (bit 1) to clear
}

void i2c_init() {
    // Assuming system clock ~12.5 MHz, set prescaler for ~100kHz I2C
    I2C_PRER_LO = 0xC7;     // Example for 100kHz (adjust based on your clk)
    I2C_PRER_HI = 0x00;
    I2C_CTR     = 0x80;     // Enable core (bit 7)
}

void i2c_start() {
    I2C_CR = 0x90; // START + WRITE
    i2c_wait();
}

void i2c_write(uint8_t val) {
    I2C_TXR = val;
    I2C_CR = 0x10;  // WRITE
    i2c_wait();
}

void i2c_stop() {
    I2C_CR = 0x40; // STOP
    i2c_wait();
}

uint8_t i2c_read(int ack) {
    I2C_CR = (ack ? 0x28 : 0x20); // READ + ACK (or NACK)
    i2c_wait();
    return I2C_RXR;
}

uint8_t ft6336u_read_chip_id() {
    i2c_start();
    i2c_write((FT6336U_ADDR << 1) | 0);     // Write mode
    i2c_write(FT6336U_CHIPID_REG);         // Chip ID register
    i2c_start();
    i2c_write((FT6336U_ADDR << 1) | 1);     // Read mode
    uint8_t chip_id = i2c_read(0);         // Read and NACK
    i2c_stop();
    return chip_id;
}



int main() {
    config_uart();    // Make sure UART is initialized
    i2c_init();     // Initialize I2C master

    delay(10000);
    ee_printf("[FT6336U] Probing chip ID...\n");

    uint8_t chip_id = ft6336u_read_chip_id();
    ee_printf("[FT6336U] Chip ID = 0x%02X\n", chip_id);

    while (1);
    return 0;
}

