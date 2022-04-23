#### [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

输入一个整数 `n` ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

 

**示例 1：**

```
输入：n = 12
输出：5
```

**示例 2：**

```
输入：n = 13
输出：6
```

 
```C++
class Solution {
public:
    int countDigitOne(int n) {
        if (n <= 0)return 0;
        string s = to_string(n);
        int top = s[0] - '0', digit = pow(10, s.size() - 1);
        int remained = n % digit;
        return (top == 1) ?
               countDigitOne(remained) + countDigitOne(digit - 1) + (remained + 1) :
               countDigitOne(remained) + countDigitOne(digit - 1) * top + digit;
    }
};
```

