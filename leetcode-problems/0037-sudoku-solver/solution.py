class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        # Initialize constraints
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    boxes[(i // 3) * 3 + (j // 3)].add(board[i][j])

        def backtrack():
            for i in range(9):
                for j in range(9):
                    if board[i][j] == ".":
                        box_id = (i // 3) * 3 + (j // 3)
                        for ch in "123456789":
                            if ch not in rows[i] and ch not in cols[j] and ch not in boxes[box_id]:
                                board[i][j] = ch
                                rows[i].add(ch)
                                cols[j].add(ch)
                                boxes[box_id].add(ch)

                                if backtrack():
                                    return True

                                # undo
                                board[i][j] = "."
                                rows[i].remove(ch)
                                cols[j].remove(ch)
                                boxes[box_id].remove(ch)
                        return False
            return True

        backtrack()

