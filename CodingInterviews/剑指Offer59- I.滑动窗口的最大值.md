#### [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode.cn/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

难度困难436收藏分享切换为英文接收动态反馈

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

**示例:**

```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

 ```C++
 class Solution {
 public:
     class MyQueue { //单调队列（从大到小）
     public:
         deque<int> que; // 使用deque来实现单调队列
         void pop(int value) {
             if (!que.empty() && value == que.front()) {
                 que.pop_front();
             }
         }
         void push(int value) {
             while (!que.empty() && value > que.back()) {
                 que.pop_back();
             }
             que.push_back(value);
 
         }
         int front() {
             return que.front();
         }
     };
     vector<int> maxSlidingWindow(vector<int>& nums, int k) {
         MyQueue que;
         vector<int> result;
         if (nums.empty()) {
             return result;
         }
         for (int i = 0; i < k; i++) { // 先将前k的元素放进队列
             que.push(nums[i]);
         }
         result.push_back(que.front()); // result 记录前k的元素的最大值
         for (int i = k; i < nums.size(); i++) {
             que.pop(nums[i - k]); // 模拟滑动窗口的移动
             que.push(nums[i]); // 模拟滑动窗口的移动
             result.push_back(que.front()); // 记录对应的最大值
         }
         return result;
     }
 };
 ```

