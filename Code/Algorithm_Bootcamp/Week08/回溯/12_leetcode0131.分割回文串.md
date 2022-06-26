#### [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。

**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```
输入：s = "a"
输出：[["a"]]
```

```C++
class Solution {
public:
    vector<vector<bool>> f;
    vector<vector<string>> ans;
    vector<string> path;

    vector<vector<string>> partition(string s) {
        int n = s.size();
        f = vector<vector<bool>>(n, vector<bool>(n));
        for (int j = 0; j < n; j ++ )
            for (int i = 0; i <= j; i ++ )
                if (i == j) f[i][j] = true;
                else if (s[i] == s[j]) {
                    if (i + 1 > j - 1 || f[i + 1][j - 1]) f[i][j] = true;
                }

        dfs(s, 0);
        return ans;
    }

    void dfs(string& s, int u) {
        if (u == s.size()) ans.push_back(path);
        else {
            for (int i = u; i < s.size(); i ++ )
                if (f[u][i]) {
                    path.push_back(s.substr(u, i - u + 1));
                    dfs(s, i + 1);
                    path.pop_back();
                }
        }
    }
};
```
