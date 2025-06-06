module i2c_wrapper (
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,
    input  wire [2:0]  wb_adr_i,
    input  wire [7:0]  wb_dat_i,
    output wire [7:0]  wb_dat_o,
    input  wire        wb_we_i,
    input  wire        wb_stb_i,
    input  wire        wb_cyc_i,
    output wire        wb_ack_o,
    inout  wire        scl_io,
    inout  wire        sda_io
);

wire scl_pad_o, scl_padoen_o, scl_pad_i;
wire sda_pad_o, sda_padoen_o, sda_pad_i;

// Map bidirectional I/O
assign scl_pad_i = scl_io;
assign scl_io    = scl_padoen_o ? 1'bz : scl_pad_o;

assign sda_pad_i = sda_io;
assign sda_io    = sda_padoen_o ? 1'bz : sda_pad_o;

// Internal 8-bit WISHBONE bus interface
i2c_master_top i2c_core (
    .wb_clk_i     (wb_clk_i),
    .wb_rst_i     (wb_rst_i),
    .wb_adr_i     (wb_adr_i),
    .wb_dat_i     (wb_dat_i),
    .wb_dat_o     (wb_dat_o),
    .wb_we_i      (wb_we_i),
    .wb_stb_i     (wb_stb_i),
    .wb_cyc_i     (wb_cyc_i),
    .wb_ack_o     (wb_ack_o),
    .scl_pad_i    (scl_pad_i),
    .scl_pad_o    (scl_pad_o),
    .scl_padoen_o (scl_padoen_o),
    .sda_pad_i    (sda_pad_i),
    .sda_pad_o    (sda_pad_o),
    .sda_padoen_o (sda_padoen_o)
);

endmodule
