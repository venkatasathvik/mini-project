`timescale 1ns / 1ps

module tb_mac_pe;

    parameter DATA_WIDTH = 8;
    parameter PSUM_WIDTH = 32;

    reg clk;
    reg reset;
    reg enable;
    reg signed [DATA_WIDTH-1:0] act_in;
    reg signed [DATA_WIDTH-1:0] weight_in;
    reg signed [PSUM_WIDTH-1:0] psum_in;

    wire signed [DATA_WIDTH-1:0] act_out;
    wire signed [PSUM_WIDTH-1:0] psum_out;

    // Instantiate the Processing Element
    mac_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .PSUM_WIDTH(PSUM_WIDTH)
    ) uut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .act_in(act_in),
        .weight_in(weight_in),
        .psum_in(psum_in),
        .act_out(act_out),
        .psum_out(psum_out)
    );

    // 100 MHz clock generation (10 ns period)
    always #5 clk = ~clk;

    integer error_count = 0;

    initial begin
        clk = 0;
        reset = 1;
        enable = 0;
        act_in = 0;
        weight_in = 0;
        psum_in = 0;

        #20;
        reset = 0;
        enable = 1;

        // Test 1: Zero Case
        @(posedge clk);
        act_in    <= 8'sd0;
        weight_in <= 8'sd0;
        psum_in   <= 32'sd0;

        @(posedge clk);
        #1;
        if (psum_out === 32'sd0 && act_out === 8'sd0)
            $display("[PASS] Case 1 (Zero): act=0, weight=0, psum=0 -> psum_out=%0d", psum_out);
        else begin
            $display("[FAIL] Case 1: Expected 0, got %0d", psum_out);
            error_count = error_count + 1;
        end

        // Test 2: Standard Case (15 * -3 + 100 = 55)
        act_in    <= 8'sd15;
        weight_in <= -8'sd3;
        psum_in   <= 32'sd100;

        @(posedge clk);
        #1;
        if (psum_out === 32'sd55 && act_out === 8'sd15)
            $display("[PASS] Case 2 (Standard): act=15, weight=-3, psum=100 -> psum_out=%0d", psum_out);
        else begin
            $display("[FAIL] Case 2: Expected 55, got %0d", psum_out);
            error_count = error_count + 1;
        end

        // Test 3: Max Positive Boundary (127 * 127 + 2147483600)
        act_in    <= 8'sd127;
        weight_in <= 8'sd127;
        psum_in   <= 32'sd2147483600;

        @(posedge clk);
        #1;
        if (psum_out === 32'sh80003ED1)
            $display("[PASS] Case 3 (Max Pos): act=127, weight=127, psum=2147483600 -> psum_out=%0d (Hex: %08h)", psum_out, psum_out);
        else begin
            $display("[FAIL] Case 3: Expected -2147467567, got %0d", psum_out);
            error_count = error_count + 1;
        end

        // Test 4: Max Negative Boundary (-128 * -128 + 0 = 16384)
        act_in    <= -8'sd128;
        weight_in <= -8'sd128;
        psum_in   <= 32'sd0;

        @(posedge clk);
        #1;
        if (psum_out === 32'sd16384 && act_out === -8'sd128)
            $display("[PASS] Case 4 (Max Neg): act=-128, weight=-128, psum=0 -> psum_out=%0d", psum_out);
        else begin
            $display("[FAIL] Case 4: Expected 16384, got %0d", psum_out);
            error_count = error_count + 1;
        end

        $display("----------------------------------------");
        if (error_count == 0)
            $display("RESULT: ALL 4 TEST CASES PASSED SUCCESSFULLY!");
        else
            $display("RESULT: %0d TEST CASE(S) FAILED.", error_count);
        $display("----------------------------------------");
        
                // Test 5: True max positive (127*127 + 2147467518 = 2147483647)
        act_in    <= 8'sd127;
        weight_in <= 8'sd127;
        psum_in   <= 32'sd2147467518;
        @(posedge clk);
        #1;
        if (psum_out === 32'sh7FFFFFFF)
            $display("[PASS] Case 5 (True Max Pos): psum_out=%0d", psum_out);
        else begin
            $display("[FAIL] Case 5: Expected 2147483647, got %0d", psum_out);
            error_count = error_count + 1;
        end

        // Test 6: enable=0 must hold outputs (still 0x7FFFFFFF / 127 from Case 5)
        enable    <= 1'b0;
        act_in    <= 8'sd5;
        weight_in <= 8'sd5;
        psum_in   <= 32'sd1000;
        @(posedge clk);
        #1;
        if (psum_out === 32'sh7FFFFFFF && act_out === 8'sd127)
            $display("[PASS] Case 6 (Enable=0 holds outputs)");
        else begin
            $display("[FAIL] Case 6: outputs changed while enable=0");
            error_count = error_count + 1;
        end

        // Test 7: reset must clear both outputs
        reset <= 1'b1;
        @(posedge clk);
        #1;
        if (psum_out === 32'sd0 && act_out === 8'sd0)
            $display("[PASS] Case 7 (Reset clears outputs)");
        else begin
            $display("[FAIL] Case 7: outputs not cleared by reset");
            error_count = error_count + 1;
        end
        #20;
        $finish;
    end

endmodule