module i2c_master (
    input  logic       clk,
    input  logic       rst,
    input  logic       start,
    input  logic       read,
    input  logic       write,
    input  logic       stop,
    input  logic [7:0] din,
    output logic [7:0] dout,
    output logic       done,
    output logic       scl,
    inout  wire        sda
);
    typedef enum logic [2:0] {
        IDLE, WAIT, START, WRITE_BIT, READ_BIT, ACK_BIT, STOP
    } state_t;

    state_t state = IDLE;
    logic [3:0] bit_cnt;
    logic [7:0] shift_reg;
    logic       sda_out_en, sda_out;
    logic [7:0] rx_reg;
    logic [7:0] delay_cnt;

    assign sda = sda_out_en ? sda_out : 1'bz;
    assign scl = (delay_cnt < 100) ? 1'b0 : 1'b1;  // crude clock stretch (slow)
    assign dout = rx_reg;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= IDLE;
            done        <= 0;
            sda_out_en  <= 0;
            sda_out     <= 1;
            bit_cnt     <= 0;
            shift_reg   <= 0;
            rx_reg      <= 0;
            delay_cnt   <= 0;
        end else begin
            delay_cnt <= delay_cnt + 1;

            case (state)
                IDLE: begin
                    delay_cnt   <= 0;
                    sda_out_en  <= 0;
                    sda_out     <= 1;
                    done        <= 0;
                    if (start)
                        state <= WAIT;
                end

                WAIT: begin
                    if (delay_cnt >= 200) begin
                        delay_cnt <= 0;
                        sda_out   <= 0;
                        sda_out_en <= 1;
                        state     <= START;
                    end
                end

                START: begin
                    shift_reg <= din;
                    bit_cnt   <= 7;
                    if (write)
                        state <= WRITE_BIT;
                    else if (read)
                        state <= READ_BIT;
                end

                WRITE_BIT: begin
                    sda_out     <= shift_reg[bit_cnt];
                    sda_out_en  <= 1;
                    if (bit_cnt == 0)
                        state <= ACK_BIT;
                    else
                        bit_cnt <= bit_cnt - 1;
                end

                READ_BIT: begin
                    sda_out_en    <= 0;
                    rx_reg[bit_cnt] <= sda;
                    if (bit_cnt == 0)
                        state <= ACK_BIT;
                    else
                        bit_cnt <= bit_cnt - 1;
                end

                ACK_BIT: begin
                    sda_out_en <= 0;
                    if (stop) begin
                        sda_out_en <= 1;
                        sda_out    <= 0;
                        state      <= STOP;
                    end else begin
                        state <= IDLE;
                        done  <= 1;
                    end
                end

                STOP: begin
                    sda_out     <= 1;
                    sda_out_en  <= 1;
                    state       <= IDLE;
                    done        <= 1;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
