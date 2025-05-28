// ======================= wb_i2c_ts.v =======================
// Wishbone wrapper around i2c_master
module wb_i2c_ts (
    input  logic        wb_clk_i,
    input  logic        wb_rst_i,

    input  logic [31:0] wb_adr_i,
    input  logic [31:0] wb_dat_i,
    input  logic  [3:0] wb_sel_i,
    input  logic        wb_we_i,
    input  logic        wb_cyc_i,
    input  logic        wb_stb_i,
    output logic [31:0] wb_dat_o,
    output logic        wb_ack_o,

    output logic        scl,
    inout  wire         sda
);
    logic [7:0] i2c_din, i2c_dout;
    logic       i2c_start, i2c_read, i2c_write, i2c_stop, i2c_done;

    logic [7:0] ctrl, status, tx_data, rx_data;
    logic [7:0] command;

    i2c_master i2c_inst (
        .clk(wb_clk_i),
        .rst(wb_rst_i),
        .start(i2c_start),
        .read(i2c_read),
        .write(i2c_write),
        .stop(i2c_stop),
        .din(tx_data),
        .dout(rx_data),
        .done(i2c_done),
        .scl(scl),
        .sda(sda)
    );

    assign wb_ack_o = wb_stb_i & wb_cyc_i;

    always_ff @(posedge wb_clk_i) begin
        if (wb_rst_i) begin
            ctrl <= 0;
            tx_data <= 0;
            rx_data <= 0;
            command <= 0;
        end else if (wb_stb_i && wb_cyc_i && wb_we_i) begin
            case (wb_adr_i[5:2])
                4'h0: ctrl     <= wb_dat_i[7:0];
                4'h1: tx_data  <= wb_dat_i[7:0];
                4'h2: command  <= wb_dat_i[7:0];
            endcase
        end else begin
            if (i2c_done)
                rx_data <= i2c_dout;
        end
    end

    always_comb begin
        wb_dat_o = 32'h0;
        i2c_start = 0;
        i2c_stop = 0;
        i2c_write = 0;
        i2c_read = 0;
        i2c_din = tx_data;

        case (wb_adr_i[5:2])
            4'h0: wb_dat_o = {24'h0, ctrl};
            4'h1: wb_dat_o = {24'h0, tx_data};
            4'h2: wb_dat_o = {24'h0, command};
            4'h3: wb_dat_o = {24'h0, rx_data};
        endcase

        case (command)
            8'h01: i2c_start = 1;
            8'h02: i2c_write = 1;
            8'h04: i2c_read = 1;
            8'h08: i2c_stop  = 1;
        endcase
    end
endmodule


