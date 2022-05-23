#### [15. 三数之和](https://leetcode.cn/problems/3sum/)



给你一个包含 `n` 个整数的数组 `nums`，判断 `nums` 中是否存在三个元素 *a，b，c ，*使得 *a + b + c =* 0 ？请你找出所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

 

**示例 1：**

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
```

**示例 2：**

```
输入：nums = []
输出：[]
```

**示例 3：**

```
输入：nums = [0]
输出：[]
```

 

```C++
class Solution {
public:
     vector<vector<int>> threeSum(vector<int>& nums) {
         std::sort(nums.begin(), nums.end());
         vector<vector<int>> result;
         int n = nums.size();
         for (int i = 0; i < n; ++i) {
             if (i - 1 >= 0 && nums[i] == nums[i - 1]) continue; // 避免a重复
             int p = i + 1;
             int q = n - 1;
             while (p < q) {
                 if (p - 1 >= i + 1 && nums[p] == nums[p - 1]) { // 避免b重复
                     p++;
                     continue;
                 }
                 if (q + 1 <= n - 1 && nums[q] == nums[q + 1]) { // 避免c重复
                     q--;
                     continue;
                 }
                 int sum = nums[p] + nums[q];
                 if (sum == -1 * nums[i]) {
                     result.push_back({nums[i], nums[p], nums[q]});
                     p++;
                     q--;
                 } else if (sum < -1 * nums[i]) {
                    p++;
                 } else {
                    q--;
                 }
             }
         }
         return result;
     }
};
```

