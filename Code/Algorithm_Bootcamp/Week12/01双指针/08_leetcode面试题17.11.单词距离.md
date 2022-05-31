#### [面试题 17.11. 单词距离](https://leetcode.cn/problems/find-closest-lcci/)

有个内含单词的超大文本文件，给定任意两个`不同的`单词，找出在这个文件中这两个单词的最短距离(相隔单词数)。如果寻找过程在这个文件中会重复多次，而每次寻找的单词不同，你能对此优化吗?

**示例：**

```
输入：words = ["I","am","a","student","from","a","university","in","a","city"], word1 = "a", word2 = "student"
输出：1
```



```C++
class Solution {
public:
    int findClosest(vector<string>& words, string word1, string word2) {
        vector<int> w1ps;
        vector<int> w2ps;
        for (int i = 0; i < words.size(); ++i) {
            if (words[i] == word1) {
                w1ps.push_back(i);
            } else if (words[i] == word2) {
                w2ps.push_back(i);
            }
        }
        int p1 = 0;
        int p2 = 0;
        int minRet = INT_MAX;
        while (p1 < w1ps.size() && p2 < w2ps.size()) {
            if (w1ps[p1] < w2ps[p2]) {
                if (w2ps[p2] - w1ps[p1] < minRet) minRet = std::min(minRet, w2ps[p2] - w1ps[p1]);
                p1++;
            } else {
            if (w1ps[p1] - w2ps[p2] < minRet) minRet = std::min(minRet, w1ps[p1] - w2ps[p2]);
                p2++;
            }
        }
        return minRet;
    }
};
```

