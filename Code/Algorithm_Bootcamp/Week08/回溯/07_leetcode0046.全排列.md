#### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**示例 2：**

```
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

**示例 3：**

```
输入：nums = [1]
输出：[[1]]
```

 

```C++
class Solution {
public:
    vector<vector<int>> result;
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> path;
        backtrack(nums, 0, path);
        return result;
    }
    void backtrack(vector<int> nums, int k, vector<int> &path) {
        if (k == nums.size()) {
           result.push_back(path);
        }
        for (int i = 0; i < nums.size(); ++i) {
            if (find(path.begin(), path.end(), nums[i]) != path.end()) {
            continue;
            }
            path.push_back(nums[i]);
            backtrack(nums, k + 1, path);
            path.pop_back();
        }
    }
};
```

