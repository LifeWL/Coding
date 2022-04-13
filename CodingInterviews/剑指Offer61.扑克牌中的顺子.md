#### [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

从**若干副扑克牌**中随机抽 `5` 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

 

**示例 1:**

```
输入: [1,2,3,4,5]
输出: True
```

 

**示例 2:**

```
输入: [0,0,1,2,5]
输出: True
```

 

```C++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        bool m[15];
        memset(m, 0, sizeof(m));
        int minValue = 14, maxValue = 0;
        for (int item : nums) {
            if (item == 0) {
                continue;
            }
            if (m[item]) {
                return false;
            }
            m[item] = true;
            minValue = min(minValue, item);
            maxValue = max(maxValue, item);            
        }
        return maxValue - minValue + 1 <= 5;
    }
};
```

