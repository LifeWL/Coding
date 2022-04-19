#### [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

```
    1  
   / \
  2   2 
 / \ / \
3  4 4  3
```


但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

```
   1 
  / \ 
 2   2 
  \   \
   3   3
```

 

**示例 1：**

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2：**

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```


```C++
class Solution
{
public:
    bool isSymmetric(TreeNode* root) 
    {
        if (root == nullptr)
            return true;
        queue<TreeNode*> queue;
        queue.push(root);
        while (!queue.empty())
        {
            int size = queue.size();
            vector<int> tempVec;
            while (size-- > 0)
            {
                TreeNode* temp = queue.front();
                queue.pop();
                if (temp->left)
                {
                    queue.push(temp->left);
                    tempVec.push_back(temp->left->val);
                }
                else
                {
                    tempVec.push_back(-1);
                }
                if (temp->right)
                {
                    queue.push(temp->right);
                    tempVec.push_back(temp->right->val);
                }
                else
                {
                    tempVec.push_back(-1);
                }
            }
            for (int i = 0; i < tempVec.size() / 2; i++)
            {
                if (tempVec[i] != tempVec[tempVec.size() - 1 - i])
                    return false;
            }
        }
        return true;
    }
};
```

