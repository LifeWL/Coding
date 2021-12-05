#### 面试题 01.08. 零矩阵
编写一种算法，若M × N矩阵中某个元素为0，则将其所在的行与列清零。

 

**示例 1：**
```
输入：
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出：
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
```
**示例 2：**
```
输入：
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出：
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
```

```c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        //列
        int n = matrix[0].size();
        //行
        int m = matrix.size();
        //定义标记
        bool flag = false;
        for (int i = 0; i < m; ++i) {
            if (!matrix[i][0]) flag = true; 
            for (int j = 1; j < n; ++j) {
                if (!matrix[i][j]) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = m -1 ; i >= 0; --i) {
            for (int j = 1; j < n; ++j) {
                if (!matrix[i][0] || !matrix[0][j]) {
                    matrix[i][j] = 0;
                }
            }
            if (flag) matrix[i][0] = 0;
        }
    }
};
```