#### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

 

示例 1：

```
输出："We%20are%20happy."
输入：s = "We are happy."
```



```c++
class Solution {
public:
    string replaceSpace(string s) {
       int count = 0, len = s.size();
       for(char c : s) {
           if (c == ' ') count++;
       } 
       s.resize(count * 2 + len); 
       for (int i = s.size() - 1, j = len - 1; j < i; i--, j--) {
           if (s[j] != ' ') s[i] = s[j];
           else {
               s[i - 2] = '%';
               s[i - 1] = '2';
               s[i] = '0';
               i -= 2; 
               }
       }
        return s;
    }
};
```
```c++
class Solution {
public:
    string replaceSpace(string s) {
        string newstring;
        for (int i = 0; i < s.size(); i++) {
            if (s != ' ') newstring += s[i];
            else newstring += "%20"
        }
        return newstring;
    }
}
```
