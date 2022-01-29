#### [面试题 16.19. 水域大小](https://leetcode-cn.com/problems/pond-sizes-lcci/)

你有一个用于表示一片土地的整数矩阵`land`，该矩阵中每个点的值代表对应地点的海拔高度。若值为0则表示水域。由垂直、水平或对角连接的水域为池塘。池塘的大小是指相连接的水域的个数。编写一个方法来计算矩阵中所有池塘的大小，返回值需要从小到大排序。

**示例：**

```
输入：
[
  [0,2,1,0],
  [0,1,0,1],
  [1,1,0,1],
  [0,1,0,1]
]
输出： [1,2,4]
```



```C++
class Solution {
public:
    int count = 0;
    int n;
    int m;
    vector<int> pondSizes(vector<vector<int>>& land) {
        n = land.size();
        m = land[0].size();
        vector<int> result;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (land[i][j] == 0) {
                    count = 0;
                    dfs(land, i, j);
                    result.push_back(count);
                }
            }
        }
        sort(result.begin(), result.end());
        return result;
    }
    void dfs(vector<vector<int>> &land, int curi, int curj) {
        count++;
        land[curi][curj] = 1;
        vector<vector<int>> dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1},
        {-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
        for (int d = 0; d < 8; ++d) {
            int newi = curi + dirs[d][0];
            int newj = curj + dirs[d][1];
            if (newi >= 0 && newi < n && newj >= 0 && newj < m&& land[newi][newj] == 0) {
                dfs(land, newi, newj);
            }
        }
    }
};
```

