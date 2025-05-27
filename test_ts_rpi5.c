#include <stdio.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#define I2C_DEV "/dev/i2c-1"
#define FT_ADDR 0x38

int main() {
    int file;
    if ((file = open(I2C_DEV, O_RDWR)) < 0) {
        perror("Failed to open I2C device");
        return 1;
    }

    if (ioctl(file, I2C_SLAVE, FT_ADDR) < 0) {
        perror("Failed to set I2C address");
        close(file);
        return 1;
    }

    printf("FT6336U Touch Tracker (press Ctrl+C to stop)\n");

    while (1) {
        uint8_t reg = 0x02;
        uint8_t tp_status;

        // Read touch point count
        if (write(file, &reg, 1) != 1) {
            perror("Write failed");
            break;
        }
        if (read(file, &tp_status, 1) != 1) {
            perror("Read touch point failed");
            break;
        }

        if (tp_status & 0x0F) {
            // Touch point exists, read coordinates
            reg = 0x03;
            uint8_t buf[6];

            if (write(file, &reg, 1) != 1) {
                perror("Write failed (coord)");
                break;
            }
            if (read(file, buf, 6) != 6) {
                perror("Read failed (coord)");
                break;
            }

            int x = ((buf[0] & 0x0F) << 8) | buf[1];
            int y = ((buf[2] & 0x0F) << 8) | buf[3];

            printf("Touch detected: x = %d, y = %d\n", x, y);
        } else {
            // No touch
            // printf("No touch\n");
        }

        usleep(50000); // 50ms delay
    }

    close(file);
    return 0;
}
