#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

 

```C++
class Solution {
public:
    vector<vector<bool>> visited;
    int h;
    int w;
    int numIslands(vector<vector<char>>& grid) {
        h = grid.size();
        w = grid[0].size();
        visited.assign(h, vector<bool>(w));
        int result = 0;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                if (visited[i][j] != true && grid[i][j] == '1') {
                    result++;
                    dfs(grid, i, j);
                }
            }
        }
        return result;
    }
    void dfs(vector<vector<char>>& grid, int i, int j) {
        vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        visited[i][j] = true;
        for (int k = 0; k < 4; ++k) {
            int newi = i + directions[k][0];
            int newj = j + directions[k][1];
            if (newi >= 0 && newi < h && newj >= 0 && newj < w
            && visited[newi][newj] == false && grid[newi][newj] == '1') {
                dfs(grid, newi, newj);
            }
        }
    }
};
```

