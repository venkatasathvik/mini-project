`timescale 1ns / 1ps

module vpg_top (
    input  wire        clk,
    input  wire        rst,

    output reg         vsync,
    output wire        pclk,
    output reg  [13:0] pixel_addr,

    input  wire [7:0]  pixel_data,
    output reg  [7:0]  pixel_out
);

    // Temporary frame size for simulation testing
    parameter FRAME_SIZE = 16;

    reg pclk_reg;

    // Pixel clock: half the frequency of clk
    assign pclk = pclk_reg;

    always @(posedge clk or posedge rst) begin
        if (rst)
            pclk_reg <= 1'b0;
        else
            pclk_reg <= ~pclk_reg;
    end

    // Pixel address counter, pixel output and VSYNC
    always @(posedge pclk or posedge rst) begin
        if (rst) begin
            pixel_addr <= 14'd0;
            pixel_out  <= 8'd0;
            vsync      <= 1'b0;
        end
        else begin
            // Forward the input pixel
            pixel_out <= pixel_data;

            if (pixel_addr == FRAME_SIZE - 1) begin
                pixel_addr <= 14'd0;
                vsync      <= 1'b1;
            end
            else begin
                pixel_addr <= pixel_addr + 1'b1;
                vsync      <= 1'b0;
            end
        end
    end

endmodule