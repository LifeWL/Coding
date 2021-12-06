#### 48. 旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

 
**示例 1：**
```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```
**示例 2：**
```
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```
**示例 3：**
```
输入：matrix = [[1]]
输出：[[1]]
```
**示例 4：**
```
输入：matrix = [[1,2],[3,4]]
输出：[[3,1],[4,2]]
```

```c++
class Solution {
public:
void rotate(vector<vector<int>>& matrix) {
int n = matrix.size();
int s1_i = 0;
int s1_j = 0;
while (n > 1) {
int s2_i = s1_i;
int s2_j = s1_j + n - 1;
int s3_i = s1_i + n - 1;
int s3_j = s1_j + n - 1;
int s4_i = s1_i + n - 1;
int s4_j = s1_j;
for (int move = 0; move <= n - 2; ++move) {
int p1_i = s1_i;
int p1_j = s1_j + move;
int p2_i = s2_i + move;
int p2_j = s2_j;
int p3_i = s3_i;
int p3_j = s3_j - move;
int p4_i = s4_i - move;
int p4_j = s4_j;
swap(matrix,
p1_i, p1_j,
p2_i, p2_j,
p3_i, p3_j,
p4_i, p4_j);
}
s1_i++;
s1_j++;
n -= 2;
}
}
void swap(vector<vector<int>>& matrix,
int p1_i, int p1_j,
int p2_i, int p2_j,
int p3_i, int p3_j,
int p4_i, int p4_j) {
int tmp = matrix[p1_i][p1_j];
matrix[p1_i][p1_j] = matrix[p4_i][p4_j];
matrix[p4_i][p4_j] = matrix[p3_i][p3_j];
matrix[p3_i][p3_j] = matrix[p2_i][p2_j];
matrix[p2_i][p2_j] = tmp;
}
```