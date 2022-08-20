#### [559. N 叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/)

给定一个 N 叉树，找到其最大深度。

最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。

N 叉树输入按层序遍历序列化表示，每组子节点由空值分隔（请参见示例）。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
输入：root = [1,null,3,2,4,null,5,6]
输出：3
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：5
```

 **C++**

```c++
class Solution {
public:
    vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
        unordered_map<string, int> hash;
        for (int i = 0; i < list1.size(); i ++ ) hash[list1[i]] = i;
        int sum = INT_MAX;
        vector<string> res;
        for (int i = 0; i < list2.size(); i ++ ) {
            string& s = list2[i];
            if (hash.count(s)) {
                int k = i + hash[s];
                if (k < sum) {
                    sum = k;
                    res = {s};
                } else if (k == sum) {
                    res.push_back(s);
                }
            }
        }
        return res;
    }
};
```
