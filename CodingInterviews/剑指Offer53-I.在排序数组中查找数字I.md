#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

统计一个数字在排序数组中出现的次数。

 

**示例 1:**

```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

**示例 2:**

```
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

 
```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(!nums.size()) return  0;
        int l = 0, r = nums.size() - 1;
        while(l < r)       
        {
            int mid = (l + r) / 2;
            if(nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        if(nums[r] != target) return 0 ;  
        int begin = r;     
        l = 0, r = nums.size() - 1;
        while(l < r)       
        {
            int mid = (l + r + 1) / 2;
            if(nums[mid] <= target) l = mid;
            else r = mid - 1;
        }
        int end = r;           
        return end - begin + 1;
    } 
};
```

