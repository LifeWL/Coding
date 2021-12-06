#### 54. 螺旋矩阵
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

 

**示例 1：**
![img](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)
```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```
**示例 2：**
![img](https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg)
```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
``` 

```c++
class Solution {
public:
vector<int> spiralOrder(vector<vector<int>> &matrix) {
vector<int> res;
if (matrix.empty()) return res;
int m = matrix.size();
int n = matrix[0].size();
int left = 0;
int right = n - 1;
int top = 0;
int bottom = m - 1;
while (left <= right && top <= bottom) {
for (int j = left; j <= right; ++j) {
res.push_back(matrix[top][j]);
}

for (int i = top + 1; i <= bottom; ++i) {
res.push_back(matrix[i][right]);
}

if (top != bottom) { 
for (int j = right - 1; j >= left; --j) {
res.push_back(matrix[bottom][j]);
}
}

if (left != right) { 
for (int i = bottom - 1; i >= top + 1; --i) {
res.push_back(matrix[i][left]);
}
}
left++;
right--;
top++;
bottom--;
}
return res;
}
};
```