#### [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

 

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
  [20,9],
  [15,7]
]
```


```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>>ans;
        dfs(root,0,ans);
        for(int i=1;i<ans.size();i+=2)reverse(begin(ans[i]),end(ans[i]));
        return ans;
    }

    void dfs(TreeNode* rt,int dep,vector<vector<int>>&ans){
        if(!rt)return;
        (dep>=ans.size())?ans.push_back(vector<int>{rt->val}):ans[dep].push_back(rt->val);
        dfs(rt->left,dep+1,ans);
        dfs(rt->right,dep+1,ans);
    }
};
```
