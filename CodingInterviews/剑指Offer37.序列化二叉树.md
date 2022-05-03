#### [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

请实现两个函数，分别用来序列化和反序列化二叉树。

你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

**提示：** 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 [LeetCode 序列化二叉树的格式](https://support.leetcode-cn.com/hc/kb/article/1194353/)。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

**示例：**

![img](https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg)

```
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]
```

```C++
class Codec {
public:
    // Encodes a tree to a single string.
    //队列层序遍历。
    string serialize(TreeNode* root) {
        string res = "";
        if(root==NULL) return "";
        //处理一般
        queue<TreeNode*> my_queue;
        my_queue.push(root);
        TreeNode * cur = new TreeNode(0);
        while(!my_queue.empty())
        {
            //记录队列里的元素长度
            int len  = my_queue.size();
            while(len--)
            {
                cur = my_queue.front();
                my_queue.pop();
                if(cur==NULL)
                {
                    res.push_back('$');
                }else
                {
                     res.append(to_string(cur->val));
                }
                res.push_back(',');
                if(cur!=NULL)
                {
                my_queue.push(cur->left);
                my_queue.push(cur->right);
                }
            }
        }
        res.pop_back();
        
        return res;  
    }
    // Decodes your encoded data to tree.
    //重建二叉树。先将节点存起来，然后再遍历给他们建立结构！
    TreeNode* deserialize(string data) {
        //处理特殊
        if(data.size()==0) return NULL;
        int len = data.size();
        int i = 0;
        vector<TreeNode*> vec;
        while(i<len)
        {
            //遇到逗号停下来。
            string str = "";
            while(i<len&&data[i]!=',')
            {
                str.push_back(data[i]);
                i++;
            }
            //新建根节点.
            if(str=="$")
            {
                TreeNode * temp = NULL;
                vec.push_back(temp); //直接存NULL也可以。
            }else{
                int temp = std::stoi(str);
                TreeNode * cur = new TreeNode(temp);
                vec.push_back(cur);
            }
            i++;
        }
        //遍历vec，构建二叉树的结构。
        int j = 1;
        for(int i=0;j<vec.size();i++)
        {
            if(vec[i]==NULL) continue;
            if(j<vec.size()) vec[i]->left = vec[j++];
            if(j<vec.size()) vec[i]->right = vec[j++];
           
        }
        return vec[0];
    }
};
```
