#### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/11/09/200px-telephone-keypad2svg.png)

 

**示例 1：**

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**示例 2：**

```
输入：digits = ""
输出：[]
```

**示例 3：**

```
输入：digits = "2"
输出：["a","b","c"]
```

 

```C++
class Solution {
public:
    vector<string> result;
    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) return {};
        vector<string> mappings(10);
        mappings[2] = "abc";
        mappings[3] = "def";
        mappings[4] = "ghi";
        mappings[5] = "jkl";
        mappings[6] = "mno";
        mappings[7] = "pqrs";
        mappings[8] = "tuv";
        mappings[9] = "wxyz";
        vector<char> path(digits.size());
        backtrack(mappings, digits, 0, path);
        return result;
    }
    void backtrack(vector<string> mappings, string digits, int k, vector<char> path) {
        if (k == digits.size()) {
            string str(path.begin(), path.end());
            result.push_back(str);
            return;
        }
        string mapping = mappings[digits[k] - '0'];
        for (int i = 0; i < mapping.size(); ++i) {
            path[k] = mapping[i];
            backtrack(mappings, digits, k + 1, path);
        }
    }
};
```

