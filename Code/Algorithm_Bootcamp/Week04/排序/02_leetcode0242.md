#### 242. 有效的字母异位词

给定两个字符串 `*s*` 和 `*t*` ，编写一个函数来判断 `*t*` 是否是 `*s*` 的字母异位词。

**注意：**若 `*s*` 和 `*t*` 中每个字符出现的次数都相同，则称 `*s*` 和 `*t*` 互为字母异位词。

**示例 1:**

```
输入: s = "anagram", t = "nagaram"
输出: true
```

**示例 2:**

```
输入: s = "rat", t = "car"
输出: false
```

**C++**

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        unordered_map<char, int> a, b;
        for (auto c: s) a[c] ++ ;
        for (auto c: t) b[c] ++ ;
        return a == b;
    }
};
```
