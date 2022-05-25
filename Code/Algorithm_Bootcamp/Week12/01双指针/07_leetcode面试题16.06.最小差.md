#### [面试题 16.06. 最小差](https://leetcode.cn/problems/smallest-difference-lcci/)

给定两个整数数组`a`和`b`，计算具有最小差绝对值的一对数值（每个数组中取一个值），并返回该对数值的差

 

**示例：**

```
输入：{1, 3, 15, 11, 2}, {23, 127, 235, 19, 8}
输出：3，即数值对(11, 8)
```

 

```C++
class Solution {
public:
	long res = LONG_MAX;
    int smallestDifference(vector<int>& a, vector<int>& b) {
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        int n = a.size();
        int m = b.size();
        int i = 0;
        int j = 0;
        while (i < n && j < m) {
            if (a[i] >= b[j]) {
                res = std::min(res, (long)a[i] - b[j]);
                j++;
            } else {
                res = std::min(res, (long)b[j] - a[i]);
                i++;
            }
        }
        return (int)res;
    }
};
```

