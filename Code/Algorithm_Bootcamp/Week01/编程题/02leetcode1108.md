#### 1108. IP 地址无效化
给你一个有效的 IPv4 地址 address，返回这个 IP 地址的无效化版本。

所谓无效化 IP 地址，其实就是用 "[.]" 代替了每个 "."。

 

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
    string defangIPaddr(string address) {
        char str[address.size() + 2 * 3 + 1];
        for (int i = 0; i < address.size; i++) {
            if (address[i] != '.') str[i] = address[i];
            else {
                str[i] = '[';
                str[i + 1] = '.';
                str[i + 2] = ']';
                i += 2;
            }
        }
        str[i] = '\0';
        return str;
    }
};
```