#### [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

 

**示例 1：**

```
输入：x = 2.00000, n = 10
输出：1024.00000
```

**示例 2：**

```
输入：x = 2.10000, n = 3
输出：9.26100
```

**示例 3：**

```
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

 

```C++
class Solution {
public:
    double myPow(double x, int n) {
        if(n == 0){
            return 1;
        }
        else if(n < 0){
            return 1 / x * myPow(1 / x, -(n + 1));
        }
        else if(n & 1){
            return x * myPow(x * x, n >> 1);
        }
        else{
            return myPow(x * x, n >> 1);
        }
        return 0;
    }
};
```

