#### [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

编写一个程序，通过填充空格来解决数独问题。

数独的解法需 **遵循如下规则**：

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。（请参考示例图）

数独部分空格内已填入了数字，空白格用 `'.'` 表示。

 

**示例：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714svg.png)

```
输入：board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
解释：输入的数独如上图所示，唯一有效的解决方案如下所示：
```
![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714_solutionsvg.png)


```C++
class Solution {
public:
    vector<vector<bool>> rows;
    vector<vector<bool>> cols;
    vector<vector<vector<bool>>> blocks;
    bool solved = false;
    void solveSudoku(vector<vector<char>>& board) {
        rows.assign(9, vector<bool>(10, false));
        cols.assign(9, vector<bool>(10, false));
        blocks.assign(3, vector<vector<bool>>(3, vector<bool>(10, false)));
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '0';
                    rows[i][num] = true;
                    cols[j][num] = true;
                    blocks[i/3][j/3][num] = true;
                }
            }
        }
        backtrack(0, 0, board);
    }
    void backtrack(int row, int col, vector<vector<char>>& board) {
        if (row == 9) {
            solved = true;
            return;
        }
        if (board[row][col] != '.') {
            int nextRow = row;
            int nextCol = col + 1;
            if (col == 8) {
                nextRow = row + 1;
                nextCol = 0;
            }
            backtrack(nextRow, nextCol, board);
            if (solved) return;
        } else {
            for (int num = 1; num <= 9; ++num) {
                if (!rows[row][num] && !cols[col][num]
                && !blocks[row/3][col/3][num]) {
                    board[row][col] = to_string(num)[0];
                    rows[row][num] = true;
                    cols[col][num] = true;
                    blocks[row/3][col/3][num] = true;
                    int nextRow = row;
                    int nextCol = col + 1;
                    if (col == 8) {
                        nextRow = row + 1;
                        nextCol = 0;
                    }
                    backtrack(nextRow, nextCol, board);
                    if (solved) return;
                    board[row][col] = '.';
                    rows[row][num] = false;
                    cols[col][num] = false;
                    blocks[row/3][col/3][num] = false;
                }
            }
        }
    }
};
```

