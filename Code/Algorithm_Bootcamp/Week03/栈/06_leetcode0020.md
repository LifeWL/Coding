#### 20. 有效的括号

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。

 

**示例 1：**

```
输入：s = "()"
输出：true
```

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```
输入：s = "(]"
输出：false
```

**示例 4：**

```
输入：s = "([)]"
输出：false
```

**示例 5：**

```
输入：s = "{[]}"
输出：true
```

 

```c++
class Solution {
public:
    bool isValid(string s) {
        int n = s.size();
        if (n %2  == 1) return false;
        stack<char> sta;
        unordered_map<char, char> pairs {
            {'}','{'},
            {']','['},
            {')','('}
        };
        for (char c : s) {
            if (pairs.count(c)) {
                if (sta.empty() || sta.top() != pairs[c]) {
                    return false;
                } 
                sta.pop();
            } else {
                sta.push(c);
            }
        }
        return sta.empty();
    }
};
```

