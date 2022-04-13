#### [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

**示例 1:**

```
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

 

```c++
class Solution {
private:
    vector<string> res;
    string s;
    char num[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    void dfs(int x, int len){
        if(x == len){
            res.push_back(s);
            return;
        }
        int start = x == 0 ? 1 : 0;
        for(int i = start; i < 10; i++){
            s.push_back(num[i]);
            dfs(x + 1, len);
            s.pop_back();
        }
    }

public:
    vector<int> printNumbers(int n) {
        for(int i = 1; i <= n; i++){
            dfs(0, i);
        }
        vector<int> ans;
        for(int i = 0; i < res.size(); i++){
            ans.push_back(stoi(res[i]));
        }
        return ans;
    }
};
```

