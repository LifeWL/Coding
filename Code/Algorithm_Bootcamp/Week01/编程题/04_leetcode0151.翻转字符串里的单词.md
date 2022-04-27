#### 151. 翻转字符串里的单词



给你一个字符串 `s` ，逐个翻转字符串中的所有 **单词** 。

**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。

请你返回一个翻转 `s` 中单词顺序并用单个空格相连的字符串。

**说明：**

- 输入字符串 `s` 可以在前面、后面或者单词间包含多余的空格。
- 翻转后单词间应当仅用一个空格分隔。
- 翻转后的字符串中不应包含额外的空格。

 

**示例 1：**

```
输入：s = "the sky is blue"
输出："blue is sky the"
```

**示例 2：**

```
输入：s = "  hello world  "
输出："world hello"
解释：输入字符串可以在前面或者后面包含多余的空格，但是翻转后的字符不能包括。
```

**示例 3：**

```
输入：s = "a good   example"
输出："example good a"
解释：如果两个单词间有多余的空格，将翻转后单词间的空格减少到只含一个。
```

**示例 4：**

```
输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"
```

**示例 5：**

```
输入：s = "Alice does not even like bob"
输出："bob like even not does Alice"
```

 

```c++
class Solution {
public:
    void reverse(string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            swap(s[i], s[j]);
        }
        return ;
    }
    void removeExtraSpaces(string& s) {
        int fastindex = 0, slowindex = 0;
        while (s.size() > 0 && fastindex < s.size() && s[fastindex] == ' ')  {
            fastindex++;
        }
        for (; fastindex < s.size(); fastindex++) {
            if (fastindex - 1 > 0 
                && s[fastindex - 1] == s[fastindex] 
                && s[fastindex - 1] == ' '){
                    continue;
            } else {   
                s[slowindex++] = s[fastindex];
            }
        }
        if (slowindex - 1 > 0 && s[slowindex - 1] == ' ')
            { s.resize(slowindex - 1); } 
        else 
            { s.resize(slowindex); }

    }
    string reverseWords(string s) {
        removeExtraSpaces(s);
        reverse(s, 0, s.size()-1);
        for (int i = 0; i < s.size(); i++) {
            int j = i;
            while (j < s.size() && s[j] != ' ') 
            { j++; }
            reverse(s, i, j - 1);
            i = j;
        }
        return s;
    }
};
```