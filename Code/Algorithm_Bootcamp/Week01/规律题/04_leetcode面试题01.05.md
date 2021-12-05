#### 面试题 01.05. 一次编辑
字符串有三种编辑操作:插入一个字符、删除一个字符或者替换一个字符。 给定两个字符串，编写一个函数判定它们是否只需要一次(或者零次)编辑。

 

**示例 1:**
```
输入: 
first = "pale"
second = "ple"
输出: True
``` 

**示例 2:**
```
输入: 
first = "pales"
second = "pal"
输出: False
```

```c++
class Solution {
public:
    bool oneEditAway(string first, string second) {
        int lf = first.length(), ls = second.length();
        if (lf > ls)
            return oneEditAway(second, first);
        if (ls - lf > 1)
            return false;
        if (lf == ls) {
            int count = 0;
            for (int i = 0; i < lf; i++) {
                if (first[i] != second[i])
                    count += 1;
            }
            return count <= 1;
        }
        int i = 0, ofs = 0;
        while (i < lf) {
            if (first[i] != second[i + ofs]) {
                if (++ofs > 1)
                    return false;
            } else {
                i += 1;
            }
        }
        return true;
    }
};
```