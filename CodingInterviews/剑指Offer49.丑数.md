#### [剑指 Offer 49. 丑数](https://leetcode.cn/problems/chou-shu-lcof/)

难度中等340收藏分享切换为英文接收动态反馈

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

 

**示例:**

```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

**说明:** 

1. `1` 是丑数。
2. `n` **不超过**1690。

```C++
class Solution {
public:
    int nthUglyNumber(int n) {
        int f[n + 1];
        f[0] = 1;
        int p2 = 0, p3 = 0, p5 = 0;
        for(int i = 1; i <= n; i++) {
            f[i] = min(f[p2] * 2, min(f[p3] * 3, f[p5] * 5));
            if(f[i] == f[p2] * 2) p2++;
            if(f[i] == f[p3] * 3) p3++;
            if(f[i] == f[p5] * 5) p5++;
        }
        return f[n - 1];
    }
};
```

