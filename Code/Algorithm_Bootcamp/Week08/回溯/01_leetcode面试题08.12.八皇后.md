#### [面试题 08.12. 八皇后](https://leetcode-cn.com/problems/eight-queens-lcci/)

设计一种算法，打印 N 皇后在 N × N 棋盘上的各种摆法，其中每个皇后都不同行、不同列，也不在对角线上。这里的“对角线”指的是所有的对角线，不只是平分整个棋盘的那两条对角线。

**注意：**本题相对原题做了扩展

**示例:**

```
 输入：4
 输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
 解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
```
```C++
class Solution {
public:
    vector<vector<string>> result;
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<char>> board(n, vector<char>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
            board[i][j] = '.';
            }
        }
        backtrack(0, board, n);
        return result;
    }
    void backtrack(int row, vector<vector<char>>& board, int n) {
        if (row == n) {
            vector<string> snapshot;
            for (int i = 0; i < n; ++i) {
                string str(board[i].begin(), board[i].end());
                snapshot.push_back(str);
            }
            result.push_back(snapshot);
            return;
        }
        for (int col = 0; col < n; ++col) {
            if (isOk(board, n, row, col)) {
                board[row][col] = 'Q';
                backtrack(row + 1, board, n);
                board[row][col] = '.';
            }
        }
    }
    bool isOk(vector<vector<char>> board, int n, int row, int col) {
        // 检查列是否有冲突
        for (int i = 0; i < n; ++i) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        // 检查右上对⻆线是否有冲突
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; --i, ++j) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        // 检查左上对⻆线是否有冲突
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; --i, --j) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
};
```
