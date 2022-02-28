#### [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

 

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```



```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (!root) return ans;
        queue<TreeNode*> q;
        q.emplace(root);
        q.emplace(nullptr);
        while (!q.empty()) {
            vector<int> tmp;
            while (q.front()) {
                TreeNode* layer = q.front();
                q.pop();
                tmp.emplace_back(layer->val);
                if (layer->left) q.emplace(layer->left);
                if (layer->right) q.emplace(layer->right);
            }
            q.pop();
            if (!q.empty()) q.emplace(nullptr);
            ans.emplace_back(tmp);
        }
        return ans;
    }
};
```

