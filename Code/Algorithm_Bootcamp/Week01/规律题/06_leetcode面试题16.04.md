#### 面试题 16.04. 井字游戏
设计一个算法，判断玩家是否赢了井字游戏。输入是一个 N x N 的数组棋盘，由字符" "，"X"和"O"组成，其中字符" "代表一个空位。

以下是井字游戏的规则：

玩家轮流将字符放入空位（" "）中。
第一个玩家总是放字符"O"，且第二个玩家总是放字符"X"。
"X"和"O"只允许放置在空位中，不允许对已放有字符的位置进行填充。
当有N个相同（且非空）的字符填充任何行、列或对角线时，游戏结束，对应该字符的玩家获胜。
当所有位置非空时，也算为游戏结束。
如果游戏结束，玩家不允许再放置字符。
如果游戏存在获胜者，就返回该游戏的获胜者使用的字符（"X"或"O"）；如果游戏以平局结束，则返回 "Draw"；如果仍会有行动（游戏未结束），则返回 "Pending"。

**示例 1：**
```
输入： board = ["O X"," XO","X O"]
输出： "X"
```
**示例 2：**
```
输入： board = ["OOX","XXO","OXO"]
输出： "Draw"
解释： 没有玩家获胜且不存在空位
```
**示例 3：**
```
输入： board = ["OOX","XXO","OX "]
输出： "Pending"
解释： 没有玩家获胜且仍存在空位
```

```c++
class Solution {
public:
string tictactoe(vector<string> &board) {
    int n = board.size();
    bool determined = false;
    for (int i = 0; i < n; ++i) {
        if (board[i][0] == ' ') continue;
        determined = true;
        for (int j = 1; j < n; ++j) {
            if (board[i][j] != board[i][0]) {
                determined = false;
                break;
            }
        }
        string res(1, board[i][0]);
        if (determined) return res;
    }
    for (int j = 0; j < n; ++j) {
        if (board[0][j] == ' ') continue;
        determined = true;
        for (int i = 1; i < n; ++i) {
            if (board[i][j] != board[0][j]) {
                determined = false;
                break;
            }
        }
        string res(1, board[0][j]);
        if (determined) return res;
    }
    if (board[0][0] != ' ') {
        int i = 1;
        int j = 1;
        determined = true;
        while (i < n && j < n) {
            if (board[i][j] != board[0][0]) {
            determined = false;
            break;
            }
        i++;
        j++;
        }
        string res(1, board[0][0]);
        if (determined) return res;
    }
    if (board[n - 1][0] != ' ') {
        int i = n - 2, j = 1;
        determined = true;
        while (i >= 0 && j < n) {
        if (board[i][j] != board[n - 1][0]) {
        determined = false;
        break;
        }
        i--, j++;
        }
        string str(1, board[n - 1][0]);
        if (determined) return str;
    }
    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < board[i].size(); ++j) {
    if (board[i][j] == ' ') return "Pending";
    }
    }
    return "Draw";
}
};
```