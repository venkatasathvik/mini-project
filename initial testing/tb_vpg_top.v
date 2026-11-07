`timescale 1ns / 1ps
module tb_vpg_top;
reg clk;
    reg rst;
    reg [7:0] pixel_data;
wire vsync;
    wire pclk;
    wire [13:0] pixel_addr;
    wire [7:0] pixel_out;
// Connect the VPG design to the testbench
    vpg_top uut (
        .clk(clk),
        .rst(rst),
        .vsync(vsync),
        .pclk(pclk),
        .pixel_addr(pixel_addr),
        .pixel_data(pixel_data),
        .pixel_out(pixel_out)
    );
// Generate a clock with a 10 ns period
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end
// Reset and test sequence
    initial begin
        rst = 1'b1;
        pixel_data = 8'h00;
#20;
        rst = 1'b0;
repeat (20) begin
            @(posedge pclk);
            pixel_data = pixel_data + 8'h10;
        end
#500;
        $finish;
    end
// Print signal values
    initial begin
        $monitor(
            "time=%0t clk=%b rst=%b pclk=%b vsync=%b addr=%0d pixel=%h",
            $time, clk, rst, pclk, vsync, pixel_addr, pixel_out
        );
    end
endmodule