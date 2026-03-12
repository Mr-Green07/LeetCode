class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # Edge Case: If 1 row or string is shorter than numRows, no zigzag is possible
        if numRows == 1 or numRows >= len(s):
            return s
            
        # Create an array of strings to represent each row
        rows = [""] * numRows
        
        current_row = 0
        direction = -1 # We start at -1 so the first step flips it to +1 (going down)
        
        for char in s:
            # Drop the character into its current row
            rows[current_row] += char
            
            # If we hit the top row or the bottom row, flip the direction!
            if current_row == 0 or current_row == numRows - 1:
                direction *= -1
                
            # Move the elevator to the next floor
            current_row += direction
            
        # Join all the rows together into one final string
        return "".join(rows)
