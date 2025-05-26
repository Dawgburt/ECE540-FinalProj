// ======================= i2c_pmod_jb.xdc =======================
## PMOD JB
#set_property -dict { PACKAGE_PIN D14   IOSTANDARD LVCMOS33 } [get_ports { scl }]; #IO_L1P_T0_AD0P_15 Sch=jb[1]
#set_property -dict { PACKAGE_PIN F16   IOSTANDARD LVCMOS33 } [get_ports { sda }]; #IO_L14N_T2_SRCC_15 Sch=jb[2]

