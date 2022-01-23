#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

给定一个可包含重复数字的序列 `nums` ，**按任意顺序** 返回所有不重复的全排列。

 

**示例 1：**

```
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

**示例 2：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

 

```C++
class Solution {
public:
    vector<vector<int>> result;
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        unordered_map<int, int> hm;
        for (int i = 0; i < nums.size(); ++i) {
            int count = 1;
            auto it = hm.find(nums[i]);
            if (it != hm.end()) {
                count += hm[nums[i]]; 
            }
            hm[nums[i]] = count;
        }
        int n = hm.size();
        vector<int> uniqueNums(n);
        vector<int> counts(n);
        int k = 0;
        for (int i = 0; i < nums.size(); ++i) {
            auto it = hm.find(nums[i]);
            if (it != hm.end()) {
                uniqueNums[k] = nums[i];
                counts[k] = hm[nums[i]];
                k++;
                hm.erase(it);
            }
        }
        vector<int> path;
        backtrack(uniqueNums, counts, 0, path, nums.size());
        return result;
    }
    void backtrack(vector<int> uniqueNums, vector<int> counts, int k, vector<int> &path, int n) {
        if (k == n) {
            result.push_back(path);
            return;
        }
        for (int i = 0; i < uniqueNums.size(); ++i) {
            if (counts[i] == 0) continue;
            path.push_back(uniqueNums[i]);
            counts[i]--;
            backtrack(uniqueNums, counts, k + 1, path, n);
            path.pop_back();
            counts[i]++;
        }
    }
};
```

