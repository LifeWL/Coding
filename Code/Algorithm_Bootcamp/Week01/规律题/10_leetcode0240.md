#### 240. 搜索二维矩阵 II
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
 

**示例 1：**
![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid2.jpg)
```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```
**示例 2：**
![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid.jpg)
```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

```c++
class Solution {
public:
bool searchMatrix(vector<vector<int>>& matrix, int target) {
int h = matrix.size();
int w = matrix[0].size();
int i = 0;
int j = w - 1;
while (i <= h - 1 && j >= 0) {
if (matrix[i][j] == target) {
return true;
}
if (matrix[i][j] > target) {
j--;
continue;
}
if (matrix[i][j] < target) {
i++;
continue;
}
}
return false;
}
};
```

```c++
class Solution {
public:
bool searchMatrix(vector<vector<int>>& matrix, int target) {
int h = matrix.size();
int w = matrix[0].size();
int i = h - 1;
int j = 0;
while (i >= 0 && j <= w - 1) {
if (matrix[i][j] == target) {
return true;
}
if (matrix[i][j] > target) {
i--;
continue;
}
if (matrix[i][j] < target) {
j++;
continue;
}
}
return false;
}
};
```