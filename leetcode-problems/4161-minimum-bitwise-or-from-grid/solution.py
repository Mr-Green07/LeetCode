class Solution:
    def minimumOR(self, grid: List[List[int]]) -> int:
        tabol = grid
        ans = (1<<18) - 1
        for bit in range(17, -1, -1):
            candidata = ans ^ (1 << bit)
            possible = True
            for row in tabol:
                row_valid = False
                for x in row: 
                    if(x | candidata) == candidata:
                        row_valid = True
                        break
                if not row_valid:
                    possible = False
                    break
            if possible:
                ans = candidata
        return ans
