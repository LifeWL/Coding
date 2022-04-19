#### [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项（即 `F(N)`）。斐波那契数列的定义如下：

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：1
```

**示例 2：**

```
输入：n = 5
输出：5
```
 

```c++
class Solution {
public:
    int fib(int n) {
        int mod = 1000000007;
        int ans[2] = {0, 1};
        if (n < 2) 
            return ans[n];
        long long num1 = 1;
        long long num2 = 0;
        long long fibN = 0;
        for (int i = 2; i <= n; ++i) {
            fibN = (num1 + num2) % mod;
            num2 = num1;
            num1 = fibN; 
        }
        return fibN % mod;
    }
};
```

