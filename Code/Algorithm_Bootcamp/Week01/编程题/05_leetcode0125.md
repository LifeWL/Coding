#### 125. 验证回文串
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

 

**示例 1:**
```
输入: "A man, a plan, a canal: Panama"
输出: true
解释："amanaplanacanalpanama" 是回文串
```
**示例 2:**
```
输入: "race a car"
输出: false
解释："raceacar" 不是回文串
``` 

```c++
class Solution {
public:
    char toLower(char c) {
        if (c >= 'a' && c <= 'z') return c; 
        return char(c) + 32;
    }

    char isAlpha(char c) {
        if (c >= 'a' && c <= 'z') return true;
        if (c >= 'A' && c <= 'Z') return true;
        if (c >= '0' && c <= '9') return true;
        return false;
    } 

    bool isPalindrome(string s) {
        int i = 0;
        int j = s.size() - 1;
        while (i < j) {
            if (!isAlpha(s[i])) {
                ++i;
                continue;
            }
            if (!isAlpha(s[j])) {
                --j;
            continue;
            }
            if (toLower(s[i]) != toLower(s[j])) {
                return false;
            } else {
                i++;
                j--;
            }
        }
        return true;
    }
};
```