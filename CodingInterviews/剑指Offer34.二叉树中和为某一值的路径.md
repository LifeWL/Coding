#### [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

难度中等310收藏分享切换为英文接收动态反馈

给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。

**叶子节点** 是指没有子节点的节点。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```
输入：root = [1,2,3], targetSum = 5
输出：[]
```

**示例 3：**

```
输入：root = [1,2], targetSum = 0
输出：[]
```
 

```C++
class Solution {
public:
    vector<int> path;
    vector<vector<int>> ans;

    void dfs(TreeNode *u,int &sum){
        if (u == nullptr) return;
        path.push_back(u->val);
        sum -= u->val;
        if(u->left == nullptr && u->right == nullptr && sum == 0) ans.push_back(path);
        dfs(u->left, sum);
        dfs(u->right, sum);
        sum += u->val;
        path.pop_back();
    }
    
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        dfs(root, target);
        return ans;
    }
};
```

