#### [面试题 16.24. 数对和](https://leetcode.cn/problems/pairs-with-sum-lcci/)

设计一个算法，找出数组中两数之和为指定值的所有整数对。一个数只能属于一个数对。

**示例 1:**

```
输入: nums = [5,6,5], target = 11
输出: [[5,6]]
```

**示例 2:**

```
输入: nums = [5,6,5,6], target = 11
输出: [[5,6],[5,6]]
```

**提示：**

- `nums.length <= 100000`

```C++
class Solution {
public:
vector<vector<int>> pairSums(vector<int> &nums, int target) {
    vector<vector<int>> results;
    if (nums.size() == 0) return results;
    std::sort(nums.begin(), nums.end());
    int i = 0;
    int j = nums.size() - 1;
    while (i < j) {    
        if (nums[i] + nums[j] == target) {
            results.push_back({nums[i], nums[j]});
            i++;
            j--;
        } else if (nums[i] + nums[j] < target) {
            i++;
        } else {
            j--;
        }
    }
    return results;
}
};
```

