#### [面试题 08.10. 颜色填充](https://leetcode-cn.com/problems/color-fill-lcci/)

编写函数，实现许多图片编辑软件都支持的「颜色填充」功能。

待填充的图像用二维数组 `image` 表示，元素为初始颜色值。初始坐标点的行坐标为 `sr` 列坐标为 `sc`。需要填充的新颜色为 `newColor` 。

「周围区域」是指颜色相同且在上、下、左、右四个方向上存在相连情况的若干元素。

请用新颜色填充初始坐标点的周围区域，并返回填充后的图像。

 

**示例：**

```
输入：
image = [[1,1,1],[1,1,0],[1,0,1]] 
sr = 1, sc = 1, newColor = 2
输出：[[2,2,2],[2,2,0],[2,0,1]]
解释: 
初始坐标点位于图像的正中间，坐标 (sr,sc)=(1,1) 。
初始坐标点周围区域上所有符合条件的像素点的颜色都被更改成 2 。
注意，右下角的像素没有更改为 2 ，因为它不属于初始坐标点的周围区域。
```

 

```C++
class Solution {
public:
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int
    newColor) {
        int n = image.size();
        int m = image[0].size();
        dfs(image, n, m, sr, sc, image[sr][sc], newColor);
        return image;
    }
    void dfs(vector<vector<int>>& image, int n, int m, int sr, int sc, int color, int newColor) {
        image[sr][sc] = newColor;
        vector<vector<int>> dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        for (int k = 0; k < 4; ++k) {
            int newr = sr + dirs[k][0];
            int newc = sc + dirs[k][1];
            if (newr < 0 || newr >= n || newc < 0 || newc >= m || image[newr][newc] !=
                color || image[newr][newc] == newColor) {
                    continue;
            }
            dfs(image, n, m, newr, newc, color, newColor);
        }
    }
};
```

