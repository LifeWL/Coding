#### [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

**示例:**

<pre><strong>输入：</strong>s = "abc"
<strong>输出：[</strong>"abc","acb","bac","bca","cab","cba"<strong>]</strong>
</pre>

```cpp
class Solution {
public:
    vector<string> res;
    string path="";
    void dfs(const string& s,vector<bool>& used,int index) 
    {
        if(path.size()==s.size())
        {
            res.emplace_back(path);
            return;
        }
        for(int i=0;i<s.size();i++)
        {
            if(used[i]==true)
            {
                continue;
            }
            if(i>0&&used[i-1]==false&&s[i]==s[i-1])
            {
                continue;
            }
            used[i]=true;
            path+=s[i];
            dfs(s,used,index+1);
            path.pop_back();
            used[i]=false;
        }
    }
    vector<string> permutation(string s) {
        if(s.size()==0) return res;
        vector<bool> used(s.size(),false);
        sort(s.begin(),s.end());
        dfs(s,used,0);
        return res;
    }
};
```
