#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明:** 叶子节点是指没有子节点的节点。

**示例：**
给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

**C++**

```c++
class solution {
public:
    int result;
    void getdepth(treenode* node, int depth) {
        result = depth > result ? depth : result;
        if (node->left == null && node->right == null) return ;
        if (node->left) { 
            depth++;    
            getdepth(node->left, depth);
            depth--;    
        }
        if (node->right) { 
            depth++;    
            getdepth(node->right, depth);
            depth--;    
        }
        return ;
    }
    int maxdepth(treenode* root) {
        result = 0;
        if (root == NULL) return result;
        getdepth(root, 1);
        return result;
    }
};
```

