#### 1108. IP 地址无效化



给你一个有效的 [IPv4](https://baike.baidu.com/item/IPv4) 地址 `address`，返回这个 IP 地址的无效化版本。

所谓无效化 IP 地址，其实就是用 `"[.]"` 代替了每个 `"."`。

 

**示例 1：**

```
输入：address = "1.1.1.1"
输出："1[.]1[.]1[.]1"
```

**示例 2：**

```
输入：address = "255.100.50.0"
输出："255[.]100[.]50[.]0"
```

 

**示例 1：**
```
输入：address = "1.1.1.1"
输出："1[.]1[.]1[.]1"
```
**示例 2：**
```
输入：address = "255.100.50.0"
输出："255[.]100[.]50[.]0"
```



```c++
class Solution {
 public:
     string defangIPaddr(string s) {
     // 栈内存, ⾃动释放
     char str[s.size() + 2 * 3 + 1];// 注意：要多留⼀个位置给结束符
     int k = 0;
     for (int i = 0; i < s.size(); ++i) {
         if (s[i] != '.') {
         	str[k++] = s[i];
         } else {
             str[k++] = '[';
             str[k++] = '.';
             str[k++] = ']';
         }
     }
     str[k] = '\0'; // 必须有结束符
     return str;
     }
};
```