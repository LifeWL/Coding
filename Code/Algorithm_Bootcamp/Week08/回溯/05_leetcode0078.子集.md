#### [78. 子集](https://leetcode-cn.com/problems/subsets/)

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

 

```C++
class Solution {
public:
    vector<vector<int>> result;
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<int> path;
        backtrack(nums, 0, path);
        return result;
    }
    void backtrack(vector<int> nums, int k, vector<int>& path) {
        if (k == nums.size()) {
        // 批注：C++不需要snapshot，push_back()内部会拷⻉⼀份数据的副本
        result.push_back(path);
        return;
    }
    backtrack(nums, k + 1, path);
    path.push_back(nums[k]);
    backtrack(nums, k + 1, path);
    path.pop_back();
    }
};
```

