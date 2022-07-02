#### [889. 根据前序和后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

返回与给定的前序和后序遍历匹配的任何二叉树。

 `pre` 和 `post` 遍历中的值是不同的正整数。

**示例：**

```
输入：pre = [1,2,4,5,3,6,7], post = [4,5,2,6,7,3,1]
输出：[1,2,3,4,5,6,7]
```

```cpp
class Solution {
public:
    unordered_map<int, int> pos;

    TreeNode* build(vector<int>& pre, vector<int>& post, int a, int b, int x, int y) {
         if (a > b) return NULL;
         auto root = new TreeNode(pre[a]);
         if (a == b) return root;
         int k = pos[pre[a + 1]];
         root->left = build(pre, post, a + 1, a + 1 + k - x, x, k);
         root->right = build(pre, post, a + 1 + k - x + 1, b, k + 1, y - 1);
         return root;
    }

    TreeNode* constructFromPrePost(vector<int>& pre, vector<int>& post) {
        int n = pre.size();
        for (int i = 0; i < n; i ++ ) pos[post[i]] = i;
        return build(pre, post, 0, n - 1, 0, n - 1);
    }
};
```
