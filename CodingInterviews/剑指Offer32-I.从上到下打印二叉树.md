#### [面试题32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

 

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回：

```
[3,9,20,15,7]
```

 
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) 
    {
        if(!root)
        {
            return {};
        }       
        vector<int> res;
        iQueue.push(root);
        while(!iQueue.empty())
        {
            auto tmp = iQueue.front();
            res.push_back(tmp->val);         
            if(tmp->left)
            {
                iQueue.push(tmp->left);
            }
            if(tmp->right)
            {
                iQueue.push(tmp->right);
            }
            iQueue.pop();
        }
        return res;
    }
private:
    queue<TreeNode*> iQueue;
};
```

