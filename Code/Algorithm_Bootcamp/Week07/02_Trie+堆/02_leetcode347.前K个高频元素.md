#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

 

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

```C++
class Solution {
public:
    struct cmp {
        bool operator() (pair<int, int> p1, pair<int, int> p2) {
            return p1.second > p2.second; // ⼩顶堆
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        if (nums.size() < k) return {};
        vector<int> result;
        priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> pq;
        unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); ++i) {
            int count = 1;
            if (map.find(nums[i]) != map.end()) {
            count += map[nums[i]];
            }
            map[nums[i]] = count;
        }
        for (auto it = map.begin(); it != map.end(); ++it) {
            if (pq.size() < k) {
                pq.push({it->first, it->second});
            } else {
                if (it->second > pq.top().second) {
                    pq.pop();
                    pq.push({it->first, it->second});
                }
            }
        }
        for (int i = 0; i < k; ++i) {
            auto top = pq.top();
            pq.pop();
            result.push_back(top.first);
        }
        return result;
    }
};
``` 