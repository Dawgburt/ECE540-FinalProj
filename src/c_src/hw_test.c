#include <stdint.h>
#include "uart.h"
#include "ee_printf.h"

//
// =====================================
//         GPIO CONTROL SECTION
// =====================================
//

// GPIO Base and Bit Definitions
#define GPIO_BASE      0x80001400
#define GPIO_REG       (*(volatile uint32_t *)GPIO_BASE)

#define CTP_RST_BIT    (1 << 0)   // gpio_out[0]
#define LCD_RST_BIT    (1 << 14)  // gpio_out[14]
#define LCD_DC_BIT     (1 << 15)  // gpio_out[15]

void gpio_set(uint32_t mask)    { GPIO_REG |= mask; }
void gpio_clear(uint32_t mask)  { GPIO_REG &= ~mask; }
void gpio_toggle(uint32_t mask) { GPIO_REG ^= mask; }

void ctp_rst_low()   { gpio_clear(CTP_RST_BIT); }
void ctp_rst_high()  { gpio_set(CTP_RST_BIT); }

void lcd_rst_low()   { gpio_clear(LCD_RST_BIT); }
void lcd_rst_high()  { gpio_set(LCD_RST_BIT); }

void lcd_dc_low()    { gpio_clear(LCD_DC_BIT); }
void lcd_dc_high()   { gpio_set(LCD_DC_BIT); }

void delay(volatile int count) {
    while (count-- > 0) __asm__ volatile ("nop");
}

//
// =====================================
//        I2C COMMUNICATION SECTION
//         (FT6336U Touch Panel)
// =====================================
//

// I2C Registers
#define I2C_BASE        0x80006000
#define I2C_PRER_LO     (*(volatile uint32_t *)(I2C_BASE + 0x00))
#define I2C_PRER_HI     (*(volatile uint32_t *)(I2C_BASE + 0x04))
#define I2C_CTR         (*(volatile uint32_t *)(I2C_BASE + 0x08))
#define I2C_TXR         (*(volatile uint32_t *)(I2C_BASE + 0x0C))
#define I2C_RXR         (*(volatile uint32_t *)(I2C_BASE + 0x0C)) // Same as TXR
#define I2C_CR          (*(volatile uint32_t *)(I2C_BASE + 0x10))
#define I2C_SR          (*(volatile uint32_t *)(I2C_BASE + 0x10)) // Same as CR

#define FT6336U_ADDR    0x38  // 7-bit I2C address

#define I2C_START       0x90
#define I2C_STOP        0x40
#define I2C_WRITE       0x10
#define I2C_READ        0x20
#define I2C_READ_NACK   0x28
#define I2C_SR_RXACK    0x80
#define I2C_SR_TIP      0x02

void i2c_wait_tip() {
    while (I2C_SR & I2C_SR_TIP);
}

void i2c_init() {
    I2C_PRER_LO = 199;  // 100 kHz I2C with 50 MHz clk
    I2C_PRER_HI = 0;
    I2C_CTR = 0x80;     // Enable core
}

uint8_t i2c_write_byte(uint8_t byte, int start, int stop) {
    I2C_TXR = byte;
    I2C_CR = (start ? I2C_START : 0) | I2C_WRITE | (stop ? I2C_STOP : 0);
    i2c_wait_tip();
    return !(I2C_SR & I2C_SR_RXACK);
}

uint8_t i2c_read_byte(int stop) {
    I2C_CR = (stop ? I2C_READ_NACK | I2C_STOP : I2C_READ);
    i2c_wait_tip();
    return I2C_RXR;
}

uint8_t ft6336u_read_reg(uint8_t reg) {
    if (!i2c_write_byte((FT6336U_ADDR << 1) | 0, 1, 0)) return 0xFF;
    if (!i2c_write_byte(reg, 0, 0)) return 0xFF;
    if (!i2c_write_byte((FT6336U_ADDR << 1) | 1, 1, 0)) return 0xFF;
    return i2c_read_byte(1);
}

//
// =====================================
//        SPI COMMUNICATION SECTION
//         (ST7796S LCD Display)
// =====================================
//

// SPI Registers
#define SPI_BASE      0x00001000
#define SPI_TX_REG    (*(volatile uint8_t  *)(SPI_BASE + 0x03))
#define SPI_STAT_REG  (*(volatile uint8_t  *)(SPI_BASE + 0x05))

#define LCD_WIDTH     240
#define LCD_HEIGHT    320

void spi_write(uint8_t data) {
    while (!(SPI_STAT_REG & 0x01)); // Wait until ready
    SPI_TX_REG = data;
}

void lcd_write_cmd(uint8_t cmd) {
    lcd_dc_low();
    spi_write(cmd);
}

void lcd_write_data(uint8_t data) {
    lcd_dc_high();
    spi_write(data);
}

void lcd_write_pixel(uint16_t color) {
    lcd_write_data(color >> 8);
    lcd_write_data(color & 0xFF);
}

void lcd_init(void) {
    lcd_rst_low();  delay(100000);
    lcd_rst_high(); delay(120000);

    lcd_write_cmd(0x11);  // Sleep out
    delay(120000);

    lcd_write_cmd(0x3A);  // Pixel format
    lcd_write_data(0x55); // 16-bit RGB565

    lcd_write_cmd(0x36);  // Memory access control
    lcd_write_data(0x48); // MX + BGR

    lcd_write_cmd(0x29);  // Display on
    delay(100000);
}

void lcd_set_address_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    lcd_write_cmd(0x2A);
    lcd_write_data(x0 >> 8); lcd_write_data(x0 & 0xFF);
    lcd_write_data(x1 >> 8); lcd_write_data(x1 & 0xFF);

    lcd_write_cmd(0x2B);
    lcd_write_data(y0 >> 8); lcd_write_data(y0 & 0xFF);
    lcd_write_data(y1 >> 8); lcd_write_data(y1 & 0xFF);

    lcd_write_cmd(0x2C); // Start memory write
}

void lcd_fill_bar(uint16_t color, int width) {
    for (int y = 0; y < LCD_HEIGHT; y++) {
        for (int x = 0; x < width; x++) {
            lcd_write_pixel(color);
        }
    }
}

void lcd_draw_repeated_bars(void) {
    const uint16_t colors[] = {
        0xF800, // Red
        0x07E0, // Green
        0x001F, // Blue
        0xFFE0, // Yellow
        0x07FF, // Cyan
        0xF81F  // Magenta
    };
    const int num_colors = sizeof(colors) / sizeof(colors[0]);
    const int bar_width = LCD_WIDTH / num_colors;

    lcd_set_address_window(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1);

    for (int i = 0; i < num_colors; ++i) {
        lcd_fill_bar(colors[i], bar_width);
    }
}

//
// =====================================
//                MAIN
// =====================================
//

int main() {
    config_uart();
    ee_printf("\r\n🚀 Initializing system...\r\n");

    // === Initialize GPIO and reset touchscreen
    ctp_rst_low();
    delay(50000);
    ctp_rst_high();
    delay(100000);

    // === Initialize I2C and read FT6336U chip ID
    i2c_init();
    delay(10000);

    uint8_t chip_id = ft6336u_read_reg(0xA3);
    ee_printf("FT6336U Chip ID = 0x%02X\r\n", chip_id);

    if (chip_id == 0x36 || chip_id == 0x64) {
        ee_printf("✅ Valid FT6336U ID detected!\r\n");
    } else {
        ee_printf("❌ Invalid Chip ID. Check wiring or power.\r\n");
    }

    // === Initialize LCD and draw pattern
    lcd_init();
    lcd_draw_repeated_bars();

    // === Periodically read number of touches
    while (1) {
        uint8_t touches = ft6336u_read_reg(0x02);
        ee_printf("Touch Points: %d\r\n", touches & 0x0F);
        delay(100000);
    }

    return 0;
}

